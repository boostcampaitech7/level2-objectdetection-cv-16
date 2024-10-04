from collections.abc import KeysView
import json

class Custom_json_parser:
    def __init__(self, mode: str, config_json_path: str):
        self.config_json_path = config_json_path
        
        with open(self.config_json_path) as f:
            self.config = json.load(f)
    
    def get_config_from_json(self) -> dict:
        return self.config
        
    def get_config_parameters(self) -> KeysView:
        return self.config.keys()
        
# if __name__=='__main__':
    
#     asd = arguments_parser(mode='train')
    
#     a = asd.parse_base_args()
    
#     print(a)

