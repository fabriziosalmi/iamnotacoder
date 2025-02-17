import logging
from typing import Any
import mitmproxy
from mitmproxy import http

# Configure logging
logging.basicConfig(
    filename="proxy.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

blocked_ip = "1.2.3.4"

class TransparentProxy:
    """
    A transparent proxy class to block requests from a specific IP address and log other requests.
    """

    def request(self, flow: http.HTTPFlow) -> None:
        """
        Handle the HTTP request.

        Args:
            flow (http.HTTPFlow): The HTTP flow object containing the request and response information.
        """
        client_ip = flow.client_address[0]  # Get client IP

        if client_ip == blocked_ip:
            logging.info(f"Blocked request from {client_ip} to {flow.request.url}")
            flow.response = http.HTTPResponse.make(
                403, b"Forbidden", {"Content-Type": "text/plain"}
            )  # Or just drop: flow.kill()
            return  # Stop processing the request

        # Log the request (optional)
        logging.info(f"Forwarding request from {client_ip} to {flow.request.url}")

        # No modification needed for transparent proxy.  mitmproxy handles forwarding.
        # You can inspect/modify flow.request here if needed.

    def response(self, flow: http.HTTPFlow) -> None:
        """
        Handle the HTTP response.

        Args:
            flow (http.HTTPFlow): The HTTP flow object containing the request and response information.
        """
        client_ip = flow.client_address[0]
        logging.info(
            f"Received response from {flow.request.url} for {client_ip} (status code: {flow.response.status_code})"
        )
        # You can inspect/modify flow.response here if needed.

addons = [TransparentProxy()]