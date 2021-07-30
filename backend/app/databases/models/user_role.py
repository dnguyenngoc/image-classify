from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, DateTime, Boolean, DECIMAL
from databases.db import Base
from sqlalchemy.orm import relationship


class UserRole(Base):
    __tablename__ = "user_role"
    id = Column(Integer, primary_key=True)
    user_name = Column(String(255), nullable=False, index=True)
    role_name = Column(String(255), nullable=False)
    role_id =  Column(Integer, ForeignKey('role.id'), nullable=False)
    create_date = Column(DateTime, nullable=False)
    update_date = Column(DateTime, nullable=True, default=None)
    role = relationship('Role', lazy = 'noload', foreign_keys=[role_id])