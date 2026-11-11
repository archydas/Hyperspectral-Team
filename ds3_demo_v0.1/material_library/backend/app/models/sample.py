from sqlalchemy import Column, String, Text, DateTime, Integer, ForeignKey
from sqlalchemy import Uuid
from sqlalchemy.sql import func
import uuid

from ..database import Base
from ..db_types import JSONType


class Sample(Base):
    __tablename__ = "samples"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    material_id = Column(Uuid(as_uuid=True), ForeignKey("materials.id", ondelete="CASCADE"), nullable=False)
    dataset_id = Column(Uuid(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(500))
    sample_number = Column(Integer)
    sample_metadata = Column("metadata", JSONType, nullable=False, default=dict)

    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
