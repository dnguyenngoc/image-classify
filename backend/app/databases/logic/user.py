from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
from fastapi.encoders import jsonable_encoder
from sqlalchemy.orm import Session, joinedload, raiseload


from databases.models.role import Role
from databases.models.user_role import UserRole
from datetime import datetime


def get_user_role_by_name(db_session: Session, user_name: str) -> UserRole:
    return db_session.query(UserRole) \
                     .filter(UserRole.user_name == user_name) \
                     .all()