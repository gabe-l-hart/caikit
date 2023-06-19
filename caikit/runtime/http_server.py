# Copyright The Caikit Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Standard
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Iterable, Optional, Type, Union, get_args, get_origin
import asyncio
import base64
import json
import re
import ssl

# Third Party
from fastapi import FastAPI, Request, Response
from grpc import StatusCode
from sse_starlette import EventSourceResponse, ServerSentEvent
from starlette.datastructures import MutableHeaders
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import Receive
import numpy as np
import pydantic
import uvicorn

# First Party
from py_to_proto.dataclass_to_proto import Annotated  # Imported here for 3.8 compat
import aconfig
import alog

# Local
from caikit.config import get_config
from caikit.core.data_model import DataBase
from caikit.core.toolkit.sync_to_async import async_wrap_iter
from caikit.runtime.server_base import RuntimeServerBase
from caikit.runtime.service_factory import ServicePackage, ServicePackageFactory
from caikit.runtime.service_generation.rpcs import CaikitRPCBase
from caikit.runtime.servicers.global_predict_servicer import GlobalPredictServicer
from caikit.runtime.types.caikit_runtime_exception import CaikitRuntimeException
import caikit.core.toolkit.logging

## Globals #####################################################################

log = alog.use_channel("HTTP")

# Registry of DM -> Pydantic model mapping to avoid errors when reusing messages
# across endpoings
PYDANTIC_REGISTRY = {}


# Mapping from GRPC codes to their corresponding HTTP codes
# CITE: https://chromium.googlesource.com/external/github.com/grpc/grpc/+/refs/tags/v1.21.4-pre1/doc/statuscodes.md
GRPC_CODE_TO_HTTP = {
    StatusCode.OK: 200,
    StatusCode.INVALID_ARGUMENT: 400,
    StatusCode.FAILED_PRECONDITION: 400,
    StatusCode.OUT_OF_RANGE: 400,
    StatusCode.UNAUTHENTICATED: 401,
    StatusCode.PERMISSION_DENIED: 403,
    StatusCode.NOT_FOUND: 404,
    StatusCode.ALREADY_EXISTS: 409,
    StatusCode.ABORTED: 409,
    StatusCode.RESOURCE_EXHAUSTED: 429,
    StatusCode.CANCELLED: 499,
    StatusCode.UNKNOWN: 500,
    StatusCode.DATA_LOSS: 500,
    StatusCode.UNIMPLEMENTED: 501,
    StatusCode.UNAVAILABLE: 501,
    StatusCode.DEADLINE_EXCEEDED: 504,
}

## RuntimeHTTPServer ###########################################################


class RuntimeHTTPServer(RuntimeServerBase):
    """An implementation of a FastAPI server that serves caikit runtimes"""

    ###############
    ## Interface ##
    ###############

    def __init__(self, tls_config_override: Optional[aconfig.Config] = None):
        super().__init__(get_config().runtime.http.port, tls_config_override)

        self.app = FastAPI()

        # Set up the central predict servicer
        inference_service = ServicePackageFactory().get_service_package(
            ServicePackageFactory.ServiceType.INFERENCE,
        )
        self.global_predict_servicer = GlobalPredictServicer(inference_service)
        self.package_name = inference_service.descriptor.full_name.rsplit(".", 1)[0]
        self.route_prefix = self.config.runtime.http.route_prefix
        if self.route_prefix[0] != "/":
            self.route_prefix = "/" + self.route_prefix

        # Bind all routes to the server
        self._bind_routes(inference_service)

        # Bind server middleware
        self._bind_middleware()

    def start(self):
        """Start the server (blocking)"""
        # Parse TLS configuration
        tls_kwargs = {}
        if (
            self.tls_config
            and self.tls_config.server.key
            and self.tls_config.server.cert
        ):
            log.info("<RUN10001905I>", "Running with TLS")
            tls_kwargs["ssl_keyfile"] = self.tls_config.server.key
            tls_kwargs["ssl_certfile"] = self.tls_config.server.cert
            if self.tls_config.client.cert:
                log.info("<RUN10001809I>", "Running with mutual TLS")
                tls_kwargs["ssl_ca_certs"] = self.tls_config.client.cert
                tls_kwargs["ssl_cert_reqs"] = ssl.CERT_REQUIRED

        # Start the server and run forever
        uvicorn.run(
            self.app,
            host="0.0.0.0",
            port=self.port,
            log_level=None,
            log_config=None,
            **tls_kwargs,
        )

    ##########
    ## Impl ##
    ##########

    def _bind_routes(self, inference_service: ServicePackage):
        """Bind all rpcs as routes to the given app"""
        for rpc in inference_service.caikit_rpcs:
            rpc_info = rpc.create_rpc_json("")
            if rpc_info["server_streaming"]:
                self._add_unary_stream_handler(rpc)
            else:
                self._add_unary_unary_handler(rpc)

    def _bind_middleware(self):
        self.app.add_middleware(
            RawRequestBodyMiddleware, api_route_prefix=self.route_prefix
        )

    def _add_unary_unary_handler(self, rpc: CaikitRPCBase):
        """Add a unary:unary request handler for this RPC signature"""
        pydantic_request = self._dataobject_to_pydantic(
            self._get_request_dataobject(rpc)
        )
        pydantic_response = self._dataobject_to_pydantic(
            self._get_response_dataobject(rpc)
        )

        @self.app.post(self._get_route(rpc))
        async def _handler(
            model_id: str, request: pydantic_request, context: Request
        ) -> pydantic_response:
            log.debug("In unary handler for %s for model %s", rpc.name, model_id)
            loop = asyncio.get_running_loop()
            request_kwargs = {
                field: getattr(request, field) for field in request.__fields__
            }
            try:
                call = partial(
                    self.global_predict_servicer.predict_model,
                    model_id=model_id,
                    request_name=rpc.request.name,
                    **request_kwargs,
                )
                return await loop.run_in_executor(None, call)
            except CaikitRuntimeException as err:
                error_code = GRPC_CODE_TO_HTTP.get(err.status_code, 500)
                error_content = {
                    "details": err.message,
                    "code": error_code,
                    "id": err.id,
                }
            except Exception as err:
                error_code = 500
                error_content = {
                    "details": f"Unhandled exception: {str(err)}",
                    "code": error_code,
                    "id": None,
                }
                log.error("<RUN51881106E>", err, exc_info=True)
            return Response(content=json.dumps(error_content), status_code=error_code)

    def _add_unary_stream_handler(self, rpc: CaikitRPCBase):
        pydantic_request = self._dataobject_to_pydantic(
            self._get_request_dataobject(rpc)
        )
        pydantic_response = self._dataobject_to_pydantic(
            self._get_response_dataobject(rpc)
        )

        @self.app.post(self._get_route(rpc), response_model=pydantic_response)
        async def _handler(
            model_id: str, request: pydantic_request, context: Request
        ) -> EventSourceResponse:
            log.debug("In streaming handler for %s", rpc.name)
            request_kwargs = {
                field: getattr(request, field) for field in request.__fields__
            }

            async def _generator() -> pydantic_response:
                try:
                    log.debug("In stream generator for %s", rpc.name)
                    async for result in async_wrap_iter(
                        self.global_predict_servicer.predict_model(
                            model_id=model_id,
                            request_name=rpc.request.name,
                            **request_kwargs,
                        )
                    ):
                        yield result
                    return
                except CaikitRuntimeException as err:
                    error_code = GRPC_CODE_TO_HTTP.get(err.status_code, 500)
                    error_content = {
                        "details": err.message,
                        "code": error_code,
                        "id": err.id,
                    }
                except Exception as err:
                    error_code = 500
                    error_content = {
                        "details": f"Unhandled exception: {str(err)}",
                        "code": error_code,
                        "id": None,
                    }
                    log.error("<RUN51881106E>", err, exc_info=True)

                # If an error occurs, yield an error response and terminate
                yield ServerSentEvent(data=json.dumps(error_content))

            return EventSourceResponse(_generator())

    def _get_route(self, rpc: CaikitRPCBase) -> str:
        """Get the REST route for this rpc"""
        if rpc.name.endswith("Predict"):
            task_name = re.sub(
                r"(?<!^)(?=[A-Z])",
                "-",
                re.sub("Task$", "", re.sub("Predict$", "", rpc.name)),
            ).lower()
            route = "/".join([self.route_prefix, "{model_id}", "task", task_name])
            return route
        raise NotImplementedError("No support for train rpcs yet!")

    def _get_request_dataobject(self, rpc: CaikitRPCBase) -> Type[DataBase]:
        """Get the dataobject request for the given rpc"""
        return DataBase.get_class_for_name(
            ".".join([self.package_name, rpc.request.name])
        )

    @staticmethod
    def _get_response_dataobject(rpc: CaikitRPCBase) -> Type[DataBase]:
        """Get the dataobject response for the given rpc"""
        origin = get_origin(rpc.return_type)
        args = get_args(rpc.return_type)
        if isinstance(origin, type) and issubclass(origin, Iterable):
            assert args and len(args) == 1
            dm_obj = args[0]
        else:
            dm_obj = rpc.return_type
        assert isinstance(dm_obj, type) and issubclass(dm_obj, DataBase)
        return dm_obj

    @classmethod
    def _get_pydantic_type(cls, field_type: type) -> type:
        """Recursive helper to get a valid pydantic type for every field type"""
        if get_origin(field_type) is Annotated:
            field_type = get_args(field_type)[0]
        if get_origin(field_type) is Union:
            return Union.__getitem__(
                tuple(
                    (
                        cls._get_pydantic_type(arg_type)
                        for arg_type in get_args(field_type)
                    )
                )
            )
        if np.issubclass_(field_type, np.integer):
            return int
        if np.issubclass_(field_type, np.floating):
            return float
        if hasattr(field_type, "__annotations__") and not issubclass(
            field_type, pydantic.BaseModel
        ):
            return cls._dataobject_to_pydantic(field_type)
        return field_type

    @classmethod
    def _dataobject_to_pydantic(
        cls, dm_class: Type[DataBase]
    ) -> Type[pydantic.BaseModel]:
        """Make a pydantic model based on the given proto message by using the data
        model class annotations to mirror as a pydantic model
        """
        if dm_class in PYDANTIC_REGISTRY:
            return PYDANTIC_REGISTRY[dm_class]

        annotations = {
            field_name: cls._get_pydantic_type(field_type)
            for field_name, field_type in dm_class.__annotations__.items()
        }
        pydantic_model = type(pydantic.BaseModel)(
            dm_class.__name__,
            (pydantic.BaseModel,),
            {
                "__annotations__": annotations,
                **{name: None for name in dm_class.__annotations__},
            },
        )
        PYDANTIC_REGISTRY[dm_class] = pydantic_model
        return pydantic_model


## Helpers #####################################################################


@dataclass
class ReceiveProxy:
    """Proxy to starlette.types.Receive.__call__ with caching first receive call.

    CITE: https://github.com/tiangolo/fastapi/discussions/8187#discussioncomment-5148059
    """

    receive: Receive
    cached_body: bytes
    _is_first_call: ClassVar[bool] = True

    async def __call__(self):
        # First call will be for getting request body => returns cached result
        if self._is_first_call:
            self._is_first_call = False
            return {
                "type": "http.request",
                "body": self.cached_body,
                "more_body": False,
            }

        return await self.receive()


class RequestWrapper(Request):
    """This class wraps a Request so that middleware can effectively mutate a
    request without needing to monkey around with the underlying request itself
    """

    def __init__(
        self,
        wrapped_request: Request,
        prefilled_body: bytes,
        prefilled_json: dict,
        header_overrides: MutableHeaders,
    ):
        # NOTE: Intentionally not calling super().__init__ since this is a proxy
        #   to an already-initialized request

        # Explicit names that match Request
        self._body = prefilled_body
        self._receive = ReceiveProxy(wrapped_request._receive, prefilled_body)

        # Attrs that will be used for overloaded functions
        self._request = wrapped_request
        self._prefilled_json = prefilled_json
        self._headers = header_overrides

    @property
    def headers(self) -> MutableHeaders:
        # DEBUG
        breakpoint()
        return self._headers

    async def json(self) -> dict:
        return self._prefilled_json

    def __getattr__(self, name: str) -> Any:
        """Forward all other attributes to the wrapped request"""
        # DEBUG
        print(f"__getattr__({name})")
        return getattr(self._request, name)


class RawRequestBodyMiddleware(BaseHTTPMiddleware):
    def __init__(self, api_route_prefix: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_route_prefix = api_route_prefix

    async def dispatch(self, request, call_next):
        # If not an API route, ignore it
        if not request.url.path.startswith(self.api_route_prefix):
            return await call_next(request)

        mimetype = request.headers.get("content-type")
        # If the request is explicitly json, pass it along unaltered
        if mimetype == "application/json":
            return await call_next(request)

        # If the request is raw text, see if it is already json formatted and if
        # not, coerce it into a json object
        if mimetype.startswith("application") or mimetype == "text/plain":
            try:
                await request.json()
                log.debug2("Valid JSON")
            except json.decoder.JSONDecodeError:
                body = await request.body()
                # If the body is valid utf8, leave it as is
                try:
                    inputs_str = body.decode("utf-8")
                    log.debug2("Valid utf-8 string")
                except UnicodeDecodeError:
                    log.debug2("Base64 encoding bytes")
                    inputs_str = base64.encodebytes(body).decode("utf-8")
                updated_json = {"inputs": inputs_str}
                updated_body = json.dumps(updated_json).encode("utf-8")
                # headers = request.headers.mutablecopy()
                # headers["content-type"] = "application/json"
                # request = RequestWrapper(request, updated_body, updated_json, headers)
                request._receive = ReceiveProxy(
                    receive=request.receive, cached_body=updated_body
                )
                request._body = updated_body
                idx = [
                    i
                    for i, (key, _) in enumerate(request.headers._list)
                    if key == b"content-type"
                ][0]
                request.headers._list[idx] = (b"content-type", b"application/json")

                # #DEBUG
                # breakpoint()
        return await call_next(request)


## Main ########################################################################


def main():
    caikit.core.toolkit.logging.configure()
    server = RuntimeHTTPServer()
    server.start()


if __name__ == "__main__":
    main()
