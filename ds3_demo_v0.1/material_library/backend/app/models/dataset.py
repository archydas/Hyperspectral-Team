from sqlalchemy import Column, String, Text, DateTime, ForeignKey
from sqlalchemy import Uuid
from sqlalchemy.sql import func
import uuid

from ..database import Base
from ..db_types import JSONType


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    library_id = Column(Uuid(as_uuid=True), ForeignKey("libraries.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(255), nullable=False)
    slug = Column(String(255), nullable=False)
    description = Column(Text)
    source_name = Column(String(500))
    source_version = Column(String(100))
    dataset_metadata = Column("metadata", JSONType, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
