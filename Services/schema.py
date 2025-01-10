from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import date
from typing import TypedDict


class StudentBio(BaseModel):
    Name: Optional[str] = None
    Email: Optional[str] = None
    Contact: Optional[str] = None
    Links: Optional[str] = None
    Joined_on: date = None
    Enrolled_for: Optional[str] = None
    Completed_at: Optional[Any] = None


class Configuration_Database(TypedDict):
    user: Optional[str]
    password: Optional[str]
    host: Optional[str]
    port: Optional[str]
    database: Optional[str]


class Understandind_question(BaseModel):
    answer: str = Field('To understand if the llm understands the schema of the database')
