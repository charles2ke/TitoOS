"""HTTP integration built on the standard library's :mod:`urllib`.

Default-deny by design: a kernel-hosted agent can only reach hosts the
operator listed when installing the integration. That keeps a misbehaving or
prompt-injected agent from turning the process into an open proxy for internal
addresses.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Mapping

from .base import Integration, normalize_allowlist, redact

#: Header names never echoed back in :meth:`HttpIntegration.describe`.
SECRET_HEADERS = ("authorization", "proxy-authorization", "cookie", "x-api-key")

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_ALLOWED_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"})


@dataclass(frozen=True)
class HttpResponse:
    """The outcome of an HTTP call, as a plain serializable record."""

    status: int
    url: str
    body: str
    headers: Mapping[str, str] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        """Parse the body as JSON."""
        return json.loads(self.body)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "url": self.url,
            "body": self.body,
            "headers": dict(self.headers),
        }


class HttpIntegration(Integration):
    """Speak HTTP to an explicit set of hosts.

    ``allowed_hosts`` is mandatory: there is no "allow everything" mode. Hosts
    are matched on the URL's hostname, case-insensitively; a leading ``"."``
    (``".example.com"``) also matches subdomains.
    """

    name = "http"
    operations = ("get", "post_json", "request", "describe")

    def __init__(
        self,
        allowed_hosts: "list[str] | tuple[str, ...] | set[str]",
        *,
        name: str | None = None,
        timeout: float = 10.0,
        headers: Mapping[str, str] | None = None,
        max_bytes: int = 1 << 20,
    ) -> None:
        super().__init__(name)
        self.allowed_hosts = normalize_allowlist(allowed_hosts, "allowed_hosts")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.headers = dict(headers or {})

    def describe(self) -> dict[str, Any]:
        """A safe summary of this integration, with credentials redacted."""
        return {
            "name": self.name,
            "allowed_hosts": sorted(self.allowed_hosts),
            "timeout": self.timeout,
            "headers": redact(self.headers, SECRET_HEADERS),
        }

    def _host_allowed(self, host: str) -> bool:
        return any(
            host == pattern
            or (pattern.startswith(".") and host.endswith(pattern))
            or host == pattern.lstrip(".")
            for pattern in self.allowed_hosts
        )

    def _check(self, url: str, method: str, operation: str) -> str:
        if method not in _ALLOWED_METHODS:
            raise self._fail(f"unsupported HTTP method: {method!r}", operation)
        try:
            parsed = urllib.parse.urlsplit(url)
        except ValueError as exc:
            raise self._fail(f"malformed URL {url!r}: {exc}", operation) from exc
        scheme = parsed.scheme.lower()
        if scheme not in _ALLOWED_SCHEMES:
            raise self._fail(
                f"unsupported URL scheme {parsed.scheme!r}; only http and https "
                "are allowed",
                operation,
            )
        host = (parsed.hostname or "").lower()
        if not host:
            raise self._fail(f"URL has no host: {url!r}", operation)
        if not self._host_allowed(host):
            raise self._fail(
                f"host {host!r} is not in the allowlist "
                f"({', '.join(sorted(self.allowed_hosts))})",
                operation,
            )
        return urllib.parse.urlunsplit(
            (scheme, parsed.netloc, parsed.path, parsed.query, "")
        )

    def get(
        self,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        """GET ``url``, optionally with query ``params``."""
        if params:
            try:
                parsed = urllib.parse.urlsplit(url)
            except ValueError as exc:
                raise self._fail(f"malformed URL {url!r}: {exc}", "get") from exc
            encoded = urllib.parse.urlencode(params)
            # Merged through urlsplit so the query lands in the query
            # component and not inside a fragment.
            query = f"{parsed.query}&{encoded}" if parsed.query else encoded
            url = urllib.parse.urlunsplit(
                (parsed.scheme, parsed.netloc, parsed.path, query, parsed.fragment)
            )
        return self.request("GET", url, headers=headers)

    def post_json(
        self,
        url: str,
        payload: Any,
        *,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        """POST ``payload`` as a JSON body."""
        merged = {"Content-Type": "application/json", **dict(headers or {})}
        return self.request(
            "POST", url, body=json.dumps(payload).encode("utf-8"), headers=merged
        )

    def request(
        self,
        method: str,
        url: str,
        *,
        body: bytes | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> HttpResponse:
        """Perform an arbitrary allowed request and return the response.

        HTTP error statuses are returned like any other response rather than
        raised: a 404 is an answer, not a broken integration. Transport
        failures and timeouts raise :class:`~titoos.integrations.base.IntegrationError`.
        """
        method = method.upper()
        # Validate before building the request so a disallowed host or a
        # non-HTTP scheme (file://, ftp://, ...) can never be opened.
        safe_url = self._check(url, method, "request")
        request = urllib.request.Request(safe_url, data=body, method=method)
        for key, value in {**self.headers, **dict(headers or {})}.items():
            request.add_header(key, value)
        opener = urllib.request.build_opener(_NoRedirectOutsideAllowlist(self))
        try:
            with opener.open(request, timeout=self.timeout) as response:  # noqa: S310 - scheme and host are checked above
                payload = response.read(self.max_bytes + 1)
                status = response.status
                final_url = response.geturl()
                response_headers = dict(response.headers.items())
        except urllib.error.HTTPError as exc:  # an answer, not a failure
            payload = exc.read(self.max_bytes + 1)
            status = exc.code
            # The error may come from the last hop of a redirect chain, so
            # report the URL that actually answered.
            final_url = getattr(exc, "url", None) or safe_url
            response_headers = dict(exc.headers.items()) if exc.headers else {}
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise self._fail(f"{method} {safe_url} failed: {exc}", "request") from exc
        if len(payload) > self.max_bytes:
            raise self._fail(
                f"{method} {safe_url} returned more than max_bytes "
                f"({self.max_bytes} bytes)",
                "request",
            )
        return HttpResponse(
            status=status,
            url=final_url,
            body=payload.decode("utf-8", errors="replace"),
            headers=response_headers,
        )


class _NoRedirectOutsideAllowlist(urllib.request.HTTPRedirectHandler):
    """Re-check the allowlist on every redirect hop.

    Without this a permitted host could bounce an agent to an arbitrary
    address, which is exactly the hole the allowlist exists to close.
    """

    def __init__(self, integration: HttpIntegration) -> None:
        self._integration = integration

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        self._integration._check(newurl, req.get_method(), "request")
        return super().redirect_request(req, fp, code, msg, headers, newurl)
