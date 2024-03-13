from pydantic import BaseModel

class ConfigBase(BaseModel):
    def clean_model_dump(self, *args, **kwargs):
        return {
            k: v
            for k, v in self.model_dump(*args, **kwargs)
            if k and v
        }