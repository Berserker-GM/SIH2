# src/ingestor/parsers/__init__.py
from .auth_parser import parse as parse_auth
from .network_parser import parse as parse_network
from .endpoint_parser import parse as parse_endpoint
from .syslog_parser import parse as parse_syslog

PARSER_MAP = {
    "auth": parse_auth,
    "network": parse_network,
    "endpoint": parse_endpoint,
    "syslog": parse_syslog,
}