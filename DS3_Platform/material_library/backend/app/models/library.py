from sqlalchemy import Column, String, Text, DateTime
from sqlalchemy import Uuid
from sqlalchemy.sql import func
import uuid

from ..database import Base
from ..db_types import JSONType


class Library(Base):
    __tablename__ = "libraries"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    # Python attribute library_metadata  <->  database column "metadata"  (`metadata` is reserved by SQLAlchemy;
    # passing metadata=... to the constructor is silently ignored - always use library_metadata=)
    library_metadata = Column("metadata", JSONType, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
