import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, Integer, DateTime,
    ForeignKey, func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, nullable=False, index=True)
    password_hash = Column(String, nullable=False)
    created_at = Column(DateTime, server_default=func.now())

    repositories = relationship("Repository", back_populates="owner")
    search_logs = relationship("SearchLog", back_populates="user")


class Repository(Base):
    __tablename__ = "repositories"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    github_url = Column(String)
    language = Column(String)
    status = Column(String, default="pending")   # pending | indexing | ready | failed
    indexed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    owner = relationship("User", back_populates="repositories")
    chunks = relationship("CodeChunk", back_populates="repository", cascade="all, delete")


class CodeChunk(Base):
    __tablename__ = "code_chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id = Column(UUID(as_uuid=True), ForeignKey("repositories.id", ondelete="CASCADE"), nullable=False)
    file_path = Column(Text, nullable=False)
    chunk_content = Column(Text, nullable=False)
    start_line = Column(Integer)
    end_line = Column(Integer)
    language = Column(String)
    embedding_id = Column(String)    # pointer into HNSW index (str of int node id)
    chunk_type = Column(String)      # function | class | block
    created_at = Column(DateTime, server_default=func.now())

    repository = relationship("Repository", back_populates="chunks")


class SearchLog(Base):
    __tablename__ = "search_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
    query = Column(Text, nullable=False)
    latency_ms = Column(Integer)
    result_count = Column(Integer)
    created_at = Column(DateTime, server_default=func.now())

    user = relationship("User", back_populates="search_logs")
