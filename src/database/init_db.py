from sqlalchemy import create_engine
from src.config import get_settings
from src.database.models import Base
from src.config import get_settings

settings = get_settings()
print("DB HOST:", settings.database.host)
print("DB PORT:", settings.database.port)


print("USING DATABASE URL:")
print(settings.database.url)

engine = create_engine(settings.database.url)

Base.metadata.create_all(engine)

print("✅ Database tables created")
