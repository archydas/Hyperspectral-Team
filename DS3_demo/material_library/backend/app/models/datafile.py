from sqlalchemy import Column, String, Text, BigInteger, DateTime, ForeignKey
from sqlalchemy import Uuid
from sqlalchemy.sql import func
import uuid

from ..database import Base
from ..db_types import JSONType


class DataFile(Base):
    __tablename__ = "data_files"

    id = Column(Uuid(as_uuid=True), primary_key=True, default=uuid.uuid4)
    dataset_id = Column(Uuid(as_uuid=True), ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(500), nullable=False)
    file_type = Column(String(100))
    file_size = Column(BigInteger)
    checksum = Column(String(128))
    storage_path = Column(Text)
    file_metadata = Column("metadata", JSONType, nullable=False, default=dict)
    uploaded_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
