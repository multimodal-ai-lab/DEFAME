"""All integrations need to be registered here."""

from .integration import RetrievalIntegration
from .search import *
from .social_media import *

# Dynamically build the registry of retrieval integrations
# Each entry is of form: "some-domain.com": integration_object
RETRIEVAL_INTEGRATIONS: dict[str, RetrievalIntegration] = {}
for obj in list(globals().values()).copy():
    if isinstance(obj, RetrievalIntegration):
        for domain in obj.domains:
            RETRIEVAL_INTEGRATIONS[domain] = obj
