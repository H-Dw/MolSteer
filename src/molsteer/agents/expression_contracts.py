"""Typed expression grammar exposed directly in model-selected tool schemas."""
from typing import Annotated, Literal
from pydantic import Field, StrictFloat, StrictInt
from .config import StrictModel


Number = StrictInt | StrictFloat


class ObservableExpression(StrictModel):
    op: Literal['observable']
    id: str = Field(min_length=1)


class ConstantExpression(StrictModel):
    op: Literal['constant']
    value: Number
    unit: Literal['dimensionless','angstrom','angstrom^3','radian','degree','kcal/mol']
    origin: str = Field(min_length=5)


class UnaryExpression(StrictModel):
    op: Literal['relu','abs','sqrt','sin','cos']
    args: list['Expression'] = Field(min_length=1,max_length=1)


class PowerExpression(StrictModel):
    op: Literal['power']
    args: list['Expression'] = Field(min_length=1,max_length=1)
    exponent: Number = Field(ge=.5,le=8)


class BinaryExpression(StrictModel):
    op: Literal['add','subtract','multiply','divide','maximum','minimum','periodic_difference']
    args: list['Expression'] = Field(min_length=2,max_length=2)


class ReductionExpression(StrictModel):
    op: Literal['sum','mean']
    args: list['Expression'] = Field(min_length=1,max_length=32)


Expression = Annotated[ObservableExpression | ConstantExpression | UnaryExpression |
    PowerExpression | BinaryExpression | ReductionExpression, Field(discriminator='op')]
for model in (UnaryExpression, PowerExpression, BinaryExpression, ReductionExpression):
    model.model_rebuild(_types_namespace={'Expression':Expression})
