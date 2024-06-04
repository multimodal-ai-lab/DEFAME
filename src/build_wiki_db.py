from safe.tools.search.wiki_dump import WikiDumpAPI
from common.shared_config import path_to_data

wiki_dump = WikiDumpAPI()
wiki_dump.build_db(path_to_data + "FEVER/wiki-raw/")
wiki_dump._build_knn()
