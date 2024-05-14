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
"""
Caikit Core Exception enum used for reporting Exception status raised in caikit core
"""

# Standard
from enum import Enum
import uuid

################################################################################
# Caikit error types cover a number of different error conditions focused on
# error conditions in code execution (but not server communication). By design,
# they do not map explicitly to any single well-known error enum since they may
# be used in many different runtime contexts. These maps provide conversions to
# and from gRPC and HTTP error code values as the two most common error enums
# for server runtimes. We intentionally use the integer values here rather than
# relying on library enums to avoid imposing dependencies on the core. Since
# these enums are well established and not likely to change, this is a
# reasonable tradeoff.
################################################################################


class GrpcStatusCode(Enum):
    """Numeric enum of gRPC status codes. This is a local redefinition of the
    enum in order to avoid adding grpc as a dependency to the core.

    Ref: https://grpc.io/docs/guides/status-codes/
    """

    OK = 0
    CANCELLED = 1
    UNKNOWN = 2
    INVALID_ARGUMENT = 3
    DEADLINE_EXCEEDED = 4
    NOT_FOUND = 5
    ALREADY_EXISTS = 6
    PERMISSION_DENIED = 7
    RESOURCE_EXHAUSTED = 8
    FAILED_PRECONDITION = 9
    ABORTED = 10
    OUT_OF_RANGE = 11
    UNIMPLEMENTED = 12
    INTERNAL = 13
    UNAVAILABLE = 14
    DATA_LOSS = 15
    UNAUTHENTICATED = 16


class HttpStatusCode(Enum):
    """Numeric enum of HTTP status codes. This is a local redefinition of the
    enum in order to avoid adding dependencies to the core.

    Ref: https://www.w3schools.com/tags/ref_httpmessages.asp
    """

    # 1XX: INFORMATION
    CONTINUE = 100
    SWITCHING_PROTOCOLS = 101
    EARLY_HINTS = 103

    # 2XX: SUCCESSFUL
    OK = 200
    CREATED = 201
    ACCEPTED = 202
    NON_AUTHORITATIVE_INFORMATION = 203
    NO_CONTENT = 204
    RESET_CONTENT = 205
    PARTIAL_CONTENT = 206

    # 3XX: REDIRECTION
    MULTIPLE_CHOICES = 300
    MOVED_PERMANENTLY = 301
    FOUND = 302
    SEE_OTHER = 303
    NOT_MODIFIED = 304
    TEMPORARY_REDIRECT = 307
    PERMANENT_REDIRECT = 308

    # 4XX: CLIENT_ERROR
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    PAYMENT_REQUIRED = 402
    FORBIDDEN = 403
    NOT_FOUND = 404
    METHOD_NOT_ALLOWED = 405
    NOT_ACCEPTABLE = 406
    PROXY_AUTHENTICATION_REQUIRED = 407
    REQUEST_TIMEOUT = 408
    CONFLICT = 409
    GONE = 410
    LENGTH_REQUIRED = 411
    PRECONDITION_FAILED = 412
    REQUEST_TOO_LARGE = 413
    REQUEST_URI_TOO_LONG = 414
    UNSUPPORTED_MEDIA_TYPE = 415
    RANGE_NOT_SATISFIABLE = 416
    EXPECTATION_FAILED = 417

    # 5XX: SERVER_ERROR
    INTERNAL_SERVER_ERROR = 500
    NOT_IMPLEMENTED = 501
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504
    HTTP_VERSION_NOT_SUPPORTED = 505
    NETWORK_AUTHENTICATION_REQUIRED = 511


class CaikitCoreStatusCode(Enum):
    """The CaikitCoreStatusCode represents the numeric enum for"""

    OK = 0
    NOT_FOUND = 1
    INVALID_ARGUMENT = 2
    CONNECTION_ERROR = 3
    UNAUTHORIZED = 4
    FORBIDDEN = 5
    UNKNOWN = 6
    FATAL = 7

    def is_ok(self) -> bool:
        return self == self.OK

    def is_user_error(self) -> bool:
        """Whether or not this error is caused by a user as opposed to a problem
        with the system
        """
        return self in [
            self.NOT_FOUND,
            self.INVALID_ARGUMENT,
            self.CONNECTION_ERROR,
            self.UNAUTHORIZED,
            self.FORBIDDEN,
        ]

    def to_grpc(self) -> GrpcStatusCode:
        """Convert the caikit value to a grpc value"""
        return CAIKIT_CORE_TO_GRPC_STATUS.get(self, GrpcStatusCode.UNKNOWN)

    def to_http(self) -> HttpStatusCode:
        """Convert the caikit value to an http value"""
        return CAIKIT_CORE_TO_HTTP_STATUS.get(
            self, HttpStatusCode.INTERNAL_SERVER_ERROR
        )

    @classmethod
    def from_grpc(cls, grpc_code: GrpcStatusCode) -> "CaikitCoreStatusCodeEnum":
        """Convert from a grpc value to a caikit value"""
        return GRPC_TO_CAIKIT_CORE_STATUS.get(grpc_code, cls.UNKNOWN)

    @classmethod
    def from_http(cls, http_code: HttpStatusCode) -> "CaikitCoreStatusCodeEnum":
        """Convert from an http value to a caikit value"""
        return HTTP_TO_CAIKIT_CORE_STATUS.get(http_code, cls.UNKNOWN)


################################################################################
# These dicts map between caikit <-> (http, grpc). Due to the simple nature of
# the caikit enum, these mappings are very lossy.
################################################################################
GRPC_TO_CAIKIT_CORE_STATUS = {
    GrpcStatusCode.OK: CaikitCoreStatusCode.OK,
    GrpcStatusCode.CANCELLED: CaikitCoreStatusCode.CONNECTION_ERROR,
    GrpcStatusCode.UNKNOWN: CaikitCoreStatusCode.UNKNOWN,
    GrpcStatusCode.INVALID_ARGUMENT: CaikitCoreStatusCode.INVALID_ARGUMENT,
    GrpcStatusCode.DEADLINE_EXCEEDED: CaikitCoreStatusCode.CONNECTION_ERROR,
    GrpcStatusCode.NOT_FOUND: CaikitCoreStatusCode.NOT_FOUND,
    GrpcStatusCode.ALREADY_EXISTS: CaikitCoreStatusCode.INVALID_ARGUMENT,
    GrpcStatusCode.PERMISSION_DENIED: CaikitCoreStatusCode.FORBIDDEN,
    GrpcStatusCode.RESOURCE_EXHAUSTED: CaikitCoreStatusCode.INVALID_ARGUMENT,
    GrpcStatusCode.FAILED_PRECONDITION: CaikitCoreStatusCode.INVALID_ARGUMENT,
    GrpcStatusCode.ABORTED: CaikitCoreStatusCode.CONNECTION_ERROR,
    GrpcStatusCode.OUT_OF_RANGE: CaikitCoreStatusCode.INVALID_ARGUMENT,
    GrpcStatusCode.UNIMPLEMENTED: CaikitCoreStatusCode.UNKNOWN,
    GrpcStatusCode.INTERNAL: CaikitCoreStatusCode.FATAL,
    GrpcStatusCode.UNAVAILABLE: CaikitCoreStatusCode.CONNECTION_ERROR,
    GrpcStatusCode.DATA_LOSS: CaikitCoreStatusCode.CONNECTION_ERROR,
    GrpcStatusCode.UNAUTHENTICATED: CaikitCoreStatusCode.UNAUTHORIZED,
}
CAIKIT_CORE_TO_GRPC_STATUS = {
    CaikitCoreStatusCode.OK: GrpcStatusCode.OK,
    CaikitCoreStatusCode.NOT_FOUND: GrpcStatusCode.NOT_FOUND,
    CaikitCoreStatusCode.INVALID_ARGUMENT: GrpcStatusCode.INVALID_ARGUMENT,
    CaikitCoreStatusCode.CONNECTION_ERROR: GrpcStatusCode.UNAVAILABLE,
    CaikitCoreStatusCode.UNAUTHORIZED: GrpcStatusCode.UNAUTHENTICATED,
    CaikitCoreStatusCode.FORBIDDEN: GrpcStatusCode.PERMISSION_DENIED,
    CaikitCoreStatusCode.UNKNOWN: GrpcStatusCode.UNKNOWN,
    CaikitCoreStatusCode.FATAL: GrpcStatusCode.INTERNAL,
}
HTTP_TO_CAIKIT_CORE_STATUS = {
    # 1XX: INFORMATION => OK
    HttpStatusCode.CONTINUE: CaikitCoreStatusCode.OK,
    HttpStatusCode.SWITCHING_PROTOCOLS: CaikitCoreStatusCode.OK,
    HttpStatusCode.EARLY_HINTS: CaikitCoreStatusCode.OK,
    # 2XX: SUCCESSFUL => OK
    HttpStatusCode.OK: CaikitCoreStatusCode.OK,
    HttpStatusCode.CREATED: CaikitCoreStatusCode.OK,
    HttpStatusCode.ACCEPTED: CaikitCoreStatusCode.OK,
    HttpStatusCode.NON_AUTHORITATIVE_INFORMATION: CaikitCoreStatusCode.OK,
    HttpStatusCode.NO_CONTENT: CaikitCoreStatusCode.OK,
    HttpStatusCode.RESET_CONTENT: CaikitCoreStatusCode.OK,
    HttpStatusCode.PARTIAL_CONTENT: CaikitCoreStatusCode.OK,
    # 3XX: REDIRECTION => OK
    HttpStatusCode.MULTIPLE_CHOICES: CaikitCoreStatusCode.OK,
    HttpStatusCode.MOVED_PERMANENTLY: CaikitCoreStatusCode.OK,
    HttpStatusCode.FOUND: CaikitCoreStatusCode.OK,
    HttpStatusCode.SEE_OTHER: CaikitCoreStatusCode.OK,
    HttpStatusCode.NOT_MODIFIED: CaikitCoreStatusCode.OK,
    HttpStatusCode.TEMPORARY_REDIRECT: CaikitCoreStatusCode.OK,
    HttpStatusCode.PERMANENT_REDIRECT: CaikitCoreStatusCode.OK,
    # 4XX: CLIENT_ERROR
    HttpStatusCode.BAD_REQUEST: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.UNAUTHORIZED: CaikitCoreStatusCode.UNAUTHORIZED,
    HttpStatusCode.PAYMENT_REQUIRED: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.FORBIDDEN: CaikitCoreStatusCode.FORBIDDEN,
    HttpStatusCode.NOT_FOUND: CaikitCoreStatusCode.NOT_FOUND,
    HttpStatusCode.METHOD_NOT_ALLOWED: CaikitCoreStatusCode.FORBIDDEN,
    HttpStatusCode.NOT_ACCEPTABLE: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.PROXY_AUTHENTICATION_REQUIRED: CaikitCoreStatusCode.UNAUTHORIZED,
    HttpStatusCode.REQUEST_TIMEOUT: CaikitCoreStatusCode.CONNECTION_ERROR,
    HttpStatusCode.CONFLICT: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.GONE: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.LENGTH_REQUIRED: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.PRECONDITION_FAILED: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.REQUEST_TOO_LARGE: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.REQUEST_URI_TOO_LONG: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.UNSUPPORTED_MEDIA_TYPE: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.RANGE_NOT_SATISFIABLE: CaikitCoreStatusCode.INVALID_ARGUMENT,
    HttpStatusCode.EXPECTATION_FAILED: CaikitCoreStatusCode.INVALID_ARGUMENT,
    # 5XX: SERVER_ERROR
    HttpStatusCode.INTERNAL_SERVER_ERROR: CaikitCoreStatusCode.FATAL,
    HttpStatusCode.NOT_IMPLEMENTED: CaikitCoreStatusCode.UNKNOWN,
    HttpStatusCode.BAD_GATEWAY: CaikitCoreStatusCode.FATAL,
    HttpStatusCode.SERVICE_UNAVAILABLE: CaikitCoreStatusCode.FATAL,
    HttpStatusCode.GATEWAY_TIMEOUT: CaikitCoreStatusCode.FATAL,
    HttpStatusCode.HTTP_VERSION_NOT_SUPPORTED: CaikitCoreStatusCode.FATAL,
    HttpStatusCode.NETWORK_AUTHENTICATION_REQUIRED: CaikitCoreStatusCode.FATAL,
}
CAIKIT_CORE_TO_HTTP_STATUS = {
    CaikitCoreStatusCode.INVALID_ARGUMENT: HttpStatusCode.BAD_REQUEST,
    CaikitCoreStatusCode.UNAUTHORIZED: HttpStatusCode.UNAUTHORIZED,
    CaikitCoreStatusCode.FORBIDDEN: HttpStatusCode.FORBIDDEN,
    CaikitCoreStatusCode.NOT_FOUND: HttpStatusCode.NOT_FOUND,
    CaikitCoreStatusCode.CONNECTION_ERROR: HttpStatusCode.REQUEST_TIMEOUT,
    CaikitCoreStatusCode.UNKNOWN: HttpStatusCode.INTERNAL_SERVER_ERROR,
    CaikitCoreStatusCode.FATAL: HttpStatusCode.INTERNAL_SERVER_ERROR,
}


class CaikitCoreException(Exception):
    """The CaikitCoreException is the central exception type for all errors
    that can be raised by core caikit functionality. Libraries which use
    caikit.core should use CaikitCoreException to centralize errors so that they
    can be handled by other caikit layers (e.g. runtime) in a uniform way.
    """

    status_code: CaikitCoreStatusCode
    message: str

    def __init__(self, status_code: CaikitCoreStatusCode, message: str) -> None:
        self.status_code = status_code
        self.message = message
        self.id = uuid.uuid4().hex
