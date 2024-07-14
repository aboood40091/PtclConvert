import enum
from itertools import chain
import math
import os
import struct
import subprocess
import sys

from typing import Union
from typing import Tuple
from typing import Dict
from typing import ByteString
from typing import Any
from typing import List
from typing import Optional

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

try:
    from typing import Self
except ImportError:
    from typing_extensions import Self

from ninTexUtils.gx2 import GX2SurfaceDim
from ninTexUtils.gx2 import GX2SurfaceFormat
from ninTexUtils.gx2 import GX2Texture
from ninTexUtils.gx2 import GX2TileMode
from ninTexUtils.gx2.gfd import GFDFile
from PIL import Image
import pygltflib
import yaml


# Edit the content inside r''
GSH_COMPILE_PATH = r'D:\cafe_sdk-2_13_01\system\bin\win64\gshCompile.exe'

FORCE_RECOMPILE_SHADERS = False


DictGeneric = Dict[str, Any]

Numeric = Union[int, float]

Vec2Numeric = Tuple[Numeric, Numeric]
Vec2Keys = Literal['x', 'y']
Vec2Dict = Dict[Vec2Keys, Numeric]

Vec3Numeric = Tuple[Numeric, Numeric, Numeric]
Vec3Keys = Literal[Vec2Keys, 'z']
Vec3Dict = Dict[Vec3Keys, Numeric]

Vec4Numeric = Tuple[Numeric, Numeric, Numeric, Numeric]
Vec4Keys = Literal[Vec3Keys, 'w']
Vec4Dict = Dict[Vec4Keys, Numeric]

Color3Numeric = Tuple[Numeric, Numeric, Numeric]
Color3Keys = Literal['r', 'g', 'b']
Color3Dict = Dict[Color3Keys, Numeric]

Color4Numeric = Tuple[Numeric, Numeric, Numeric, Numeric]
Color4Keys = Literal[Color3Keys, 'a']
Color4Dict = Dict[Color4Keys, Numeric]

Mtx34Numeric = Tuple[Vec4Numeric, Vec4Numeric, Vec4Numeric]

UvNumeric = Tuple[Numeric, Numeric]
UvKeys = Literal['u', 'v']
UvDict = Dict[UvKeys, Numeric]


F32_STRUCT = struct.Struct('>f')


def F32Standardize(x: Numeric) -> float:
    return F32_STRUCT.unpack(F32_STRUCT.pack(x))[0]


MATH_PI = 3.141592653589793
MATH_PI_STD = F32Standardize(MATH_PI)

MATH_PI_2 = 6.283185307179586
MATH_PI_2_STD = F32Standardize(MATH_PI_2)

ENABLE_MIP_LEVEL_MAX = 15.99
ENABLE_MIP_LEVEL_MAX_STD = F32Standardize(ENABLE_MIP_LEVEL_MAX)

MIP_MAP_BIAS_MAX = 31.99
MIP_MAP_BIAS_MAX_STD = F32Standardize(MIP_MAP_BIAS_MAX)


def F32StandardToRegular(x: Numeric, std: float, reg: Numeric) -> Numeric:
    return reg if x == std else x


def F32StandardToRegularMulti(x: Numeric, std1: float, reg1: Numeric, std2: float, reg2: Numeric) -> Numeric:
    return reg1 if x == std1 else (reg2 if x == std2 else x)


def readString(data: ByteString, pos: int = 0, allow_empty: bool = True) -> str:
    end = data.find(b'\0', pos)
    if end == -1:
        raise ValueError("String must be null-terminated, but is not.")
    elif end == pos:
        if allow_empty:
            return ''
        else:
            raise ValueError("String cannot be empty, but is.")
    else:
        assert end > pos
    return data[pos:end].decode('shift_jis')


def align(x: int, y: int) -> int:
    return (x + (y - 1)) // y * y


def Vec2ToDict(vec: Vec2Numeric) -> Vec2Dict:
    return {
        "x": vec[0],
        "y": vec[1]
    }


def Vec3ToDict(vec: Vec3Numeric) -> Vec3Dict:
    return {
        "x": vec[0],
        "y": vec[1],
        "z": vec[2]
    }


def Vec4ToDict(vec: Vec4Numeric) -> Vec4Dict:
    return {
        "x": vec[0],
        "y": vec[1],
        "z": vec[2],
        "w": vec[3]
    }


def Color3ToDict(color: Color3Numeric) -> Color3Dict:
    return {
        "r": color[0],
        "g": color[1],
        "b": color[2]
    }


def UvToDict(uv: UvNumeric) -> UvDict:
    return {
        "u": uv[0],
        "v": uv[1]
    }


def S32ToHexString(x: int) -> str:
    if x >= 0:
        return '0x%08X' % x

    return '-0x%08X' % -x


def U32ToHexString(x: int) -> str:
    return '0x%08X' % x


def VerifyF32(x: Any) -> float:
    return F32Standardize(x)


def VerifyF32Positive(x: Any) -> float:
    y = F32Standardize(x)
    assert y >= 0
    return y


def VerifyF32PositiveNonZero(x: Any) -> float:
    y = F32Standardize(x)
    assert y > 0
    return y


def VerifyF32Range(x: Any, minX: Numeric, maxX: Numeric) -> float:
    assert minX <= x <= maxX
    return min(max(F32Standardize(x), minX), maxX)


def VerifyF32Normal(x: Any) -> float:
    return VerifyF32Range(x, 0, 1)


def VerifyBool(x: Any) -> bool:
    return bool(x)


def VerifyIntRange(x: Any, minX: int, maxX: int) -> int:
    assert isinstance(x, int) and minX <= x <= maxX
    return x


def VerifyU8(x: Any) -> int:
    return VerifyIntRange(x, 0, 0xFF)


def VerifyU8PositiveNonZero(x: Any) -> int:
    return VerifyIntRange(x, 1, 0xFF)


def VerifyS8(x: Any) -> int:
    return VerifyIntRange(x, -0x80, 0x7F)


def VerifyS8Positive(x: Any) -> int:
    return VerifyIntRange(x, 0, 0x7F)


def VerifyS8PositiveNonZero(x: Any) -> int:
    return VerifyIntRange(x, 1, 0x7F)


def VerifyU16(x: Any) -> int:
    return VerifyIntRange(x, 0, 0xFFFF)


def VerifyU16PositiveNonZero(x: Any) -> int:
    return VerifyIntRange(x, 1, 0xFFFF)


def VerifyS16(x: Any) -> int:
    return VerifyIntRange(x, -0x8000, 0x7FFF)


def VerifyS16Positive(x: Any) -> int:
    return VerifyIntRange(x, 0, 0x7FFF)


def VerifyS16PositiveNonZero(x: Any) -> int:
    return VerifyIntRange(x, 1, 0x7FFF)


def VerifyU32(x: Any) -> int:
    return VerifyIntRange(x, 0, 0xFFFFFFFF)


def VerifyU32PositiveNonZero(x: Any) -> int:
    return VerifyIntRange(x, 1, 0xFFFFFFFF)


def VerifyS32(x: Any) -> int:
    return VerifyIntRange(x, -0x80000000, 0x7FFFFFFF)


def VerifyS32Positive(x: Any) -> int:
    return VerifyIntRange(x, 0, 0x7FFFFFFF)


def VerifyS32PositiveNonZero(x: Any) -> int:
    return VerifyIntRange(x, 1, 0x7FFFFFFF)


def VerifyNullableStr(x: Any) -> str:
    if x is None:
        return ''

    assert isinstance(x, str)
    return x


def Vec2FromDict(dic: Vec2Dict) -> Vec2Numeric:
    return (
        VerifyF32(dic["x"]),
        VerifyF32(dic["y"])
    )


def Vec2PositiveFromDict(dic: Vec2Dict) -> Vec2Numeric:
    return (
        VerifyF32Positive(dic["x"]),
        VerifyF32Positive(dic["y"])
    )


def Vec3FromDict(dic: Vec3Dict) -> Vec3Numeric:
    return (
        VerifyF32(dic["x"]),
        VerifyF32(dic["y"]),
        VerifyF32(dic["z"])
    )


def Vec3RangeFromDict(dic: Vec3Dict, minV: Numeric, maxV: Numeric) -> Vec3Numeric:
    return (
        VerifyF32Range(dic["x"], minV, maxV),
        VerifyF32Range(dic["y"], minV, maxV),
        VerifyF32Range(dic["z"], minV, maxV)
    )


def Vec3PositiveFromDict(dic: Vec3Dict) -> Vec3Numeric:
    return (
        VerifyF32Positive(dic["x"]),
        VerifyF32Positive(dic["y"]),
        VerifyF32Positive(dic["z"])
    )


def Color3FromDict(dic: Color3Dict) -> Color3Numeric:
    return (
        VerifyF32(dic["r"]),
        VerifyF32(dic["g"]),
        VerifyF32(dic["b"])
    )


def Color3PositiveFromDict(dic: Color3Dict) -> Color3Numeric:
    return (
        VerifyF32Positive(dic["r"]),
        VerifyF32Positive(dic["g"]),
        VerifyF32Positive(dic["b"])
    )


def UvFromDict(dic: UvDict) -> UvNumeric:
    return (
        VerifyF32(dic["u"]),
        VerifyF32(dic["v"])
    )


def UvPositiveFromDict(dic: UvDict) -> UvNumeric:
    return (
        VerifyF32Positive(dic["u"]),
        VerifyF32Positive(dic["v"])
    )


def Mtx34MakeSRT(scale: Vec3Numeric, rotate: Vec3Numeric, translate: Vec3Numeric) -> Mtx34Numeric:
    f_sin = math.sin
    f_cos = math.cos
    rotateX = rotate[0]
    rotateY = rotate[1]
    rotateZ = rotate[2]
    sinRX = f_sin(rotateX)
    cosRX = f_cos(rotateX)
    sinRY = f_sin(rotateY)
    cosRY = f_cos(rotateY)
    sinRZ = f_sin(rotateZ)
    cosRZ = f_cos(rotateZ)

    scaleX = scale[0]
    scaleY = scale[1]
    scaleZ = scale[2]

    return (
        (scaleX * (cosRY * cosRZ),
         scaleY * (sinRX * sinRY * cosRZ - cosRX * sinRZ),
         scaleZ * (cosRX * cosRZ * sinRY + sinRX * sinRZ),
         translate[0]),
        (scaleX * (cosRY * sinRZ),
         scaleY * (sinRX * sinRY * sinRZ + cosRX * cosRZ),
         scaleZ * (cosRX * sinRZ * sinRY - sinRX * cosRZ),
         translate[1]),
        (scaleX * -sinRY,
         scaleY * (sinRX * cosRY),
         scaleZ * (cosRX * cosRY),
         translate[2])
    )


def Mtx34MakeRT(rotate: Vec3Numeric, translate: Vec3Numeric) -> Mtx34Numeric:
    f_sin = math.sin
    f_cos = math.cos
    rotateX = rotate[0]
    rotateY = rotate[1]
    rotateZ = rotate[2]
    sinRX = f_sin(rotateX)
    cosRX = f_cos(rotateX)
    sinRY = f_sin(rotateY)
    cosRY = f_cos(rotateY)
    sinRZ = f_sin(rotateZ)
    cosRZ = f_cos(rotateZ)

    return (
        ((cosRY * cosRZ),
         (sinRX * sinRY * cosRZ - cosRX * sinRZ),
         (cosRX * cosRZ * sinRY + sinRX * sinRZ),
         translate[0]),
        ((cosRY * sinRZ),
         (sinRX * sinRY * sinRZ + cosRX * cosRZ),
         (cosRX * sinRZ * sinRY - sinRX * cosRZ),
         translate[1]),
        (-sinRY,
         (sinRX * cosRY),
         (cosRX * cosRY),
         translate[2])
    )


S32_FMT = '>i'
S32_SIZE = struct.calcsize(S32_FMT); assert S32_SIZE == 4

U32_FMT = '>I'
U32_SIZE = struct.calcsize(U32_FMT); assert U32_SIZE == 4

F32_FMT = '>f'
F32_SIZE = struct.calcsize(F32_FMT); assert F32_SIZE == 4

VEC2_FMT = '>2f'
VEC2_SIZE = struct.calcsize(VEC2_FMT); assert VEC2_SIZE == 8

VEC3_FMT = '>3f'
VEC3_SIZE = struct.calcsize(VEC3_FMT); assert VEC3_SIZE == 0xC

VEC4_FMT = '>4f'
VEC4_SIZE = struct.calcsize(VEC4_FMT); assert VEC4_SIZE == 0x10

U32_FMT_LE = '<I'
assert struct.calcsize(U32_FMT_LE) == U32_SIZE

VEC2_FMT_LE = '<2f'
assert struct.calcsize(VEC2_FMT_LE) == VEC2_SIZE

VEC3_FMT_LE = '<3f'
assert struct.calcsize(VEC3_FMT_LE) == VEC3_SIZE

VEC4_FMT_LE = '<4f'
assert struct.calcsize(VEC4_FMT_LE) == VEC4_SIZE


class SystemConstants(enum.IntEnum):
    BinaryVersion       = 0x00000028    # EFT_BINARY_VERSION (Note: this version of PTCL does not use the geometry shader yet, and so we make sure to always check that it's disabled.)
    TexturePatternNum   = 32            # EFT_TEXTURE_PATTERN_NUM
    InfiniteLife        = 0x7fffffff    # EFT_INFINIT_LIFE


class TextureSlot(enum.IntEnum):
    FirstTexture    = 0  # EFT_TEXTURE_SLOT_0
    SecondTexture   = 1  # EFT_TEXTURE_SLOT_1
    BinMax          = 2  # EFT_TEXTURE_SLOT_BIN_MAX


class UserDataParamIdx(enum.IntEnum):
    Param_0 = 0  # EFT_USER_DATA_PARAM_0
    Param_1 = 1  # EFT_USER_DATA_PARAM_1
    Param_2 = 2  # EFT_USER_DATA_PARAM_2
    Param_3 = 3  # EFT_USER_DATA_PARAM_3
    Param_4 = 4  # EFT_USER_DATA_PARAM_4
    Param_5 = 5  # EFT_USER_DATA_PARAM_5
    Param_6 = 6  # EFT_USER_DATA_PARAM_6
    Param_7 = 7  # EFT_USER_DATA_PARAM_7
    Max     = 8  # EFT_USER_DATA_PARAM_MAX


class EmitterType(enum.IntEnum):
    Simple  = 0  # EFT_EMITTER_TYPE_SIMPLE
    Complex = 1  # EFT_EMITTER_TYPE_COMPLEX


class ColorKind(enum.IntEnum):
    Color0  = 0  # EFT_COLOR_KIND_0
    Color1  = 1  # EFT_COLOR_KIND_1
    Max     = 2  # EFT_COLOR_KIND_MAX


class ColorCalcType(enum.IntEnum):
    Fixed       = 0  # EFT_COLOR_CALC_TYPE_NONE
    Random      = 1  # EFT_COLOR_CALC_TYPE_RANDOM
    Animation   = 2  # EFT_COLOR_CALC_TYPE_RANDOM_LINEAR3COLOR


class TextureFormat(enum.IntEnum):
    Invalid     =  0  # EFT_TEXTURE_FORMAT_NONE
    Unorm_RGB8  =  1  # EFT_TEXTURE_FORMAT_24BIT_COLOR
    Unorm_RGBA8 =  2  # EFT_TEXTURE_FORMAT_32BIT_COLOR
    Unorm_BC1   =  3  # EFT_TEXTURE_FORMAT_UNORM_BC1
    SRGB_BC1    =  4  # EFT_TEXTURE_FORMAT_SRGB_BC1
    Unorm_BC2   =  5  # EFT_TEXTURE_FORMAT_UNORM_BC2
    SRGB_BC2    =  6  # EFT_TEXTURE_FORMAT_SRGB_BC2
    Unorm_BC3   =  7  # EFT_TEXTURE_FORMAT_UNORM_BC3
    SRGB_BC3    =  8  # EFT_TEXTURE_FORMAT_SRGB_BC3
    Unorm_BC4   =  9  # EFT_TEXTURE_FORMAT_UNORM_BC4
    Snorm_BC4   = 10  # EFT_TEXTURE_FORMAT_SNORM_BC4
    Unorm_BC5   = 11  # EFT_TEXTURE_FORMAT_UNORM_BC5
    Snorm_BC5   = 12  # EFT_TEXTURE_FORMAT_SNORM_BC5
    Unorm_R8    = 13  # EFT_TEXTURE_FORMAT_UNORM_8
    Unorm_RG8   = 14  # EFT_TEXTURE_FORMAT_UNORM_8_8
    SRGB_RGBA8  = 15  # EFT_TEXTURE_FORMAT_SRGB_8_8_8_8


textureFormatTable = (
  # GX2SurfaceFormat.Unorm_RGBA8,  # Skip first two entries
  # GX2SurfaceFormat.Unorm_RGBA8,
    GX2SurfaceFormat.Unorm_RGBA8,
    GX2SurfaceFormat.Unorm_BC1,
    GX2SurfaceFormat.SRGB_BC1,
    GX2SurfaceFormat.Unorm_BC2,
    GX2SurfaceFormat.SRGB_BC2,
    GX2SurfaceFormat.Unorm_BC3,
    GX2SurfaceFormat.SRGB_BC3,
    GX2SurfaceFormat.Unorm_BC4,
    GX2SurfaceFormat.Snorm_BC4,
    GX2SurfaceFormat.Unorm_BC5,
    GX2SurfaceFormat.Snorm_BC5,
    GX2SurfaceFormat.Unorm_R8,
    GX2SurfaceFormat.Unorm_RG8,
    GX2SurfaceFormat.SRGB_RGBA8
)


class ChildFlg(enum.IntFlag):
    Enable                  = ( 1 <<  0 )  # EFT_CHILD_FLAG_ENABLE
    InheritColor0           = ( 1 <<  1 )  # EFT_CHILD_FLAG_COLOR0_INHERIT
    InheritAlpha            = ( 1 <<  2 )  # EFT_CHILD_FLAG_ALPHA_INHERIT
    InheritScale            = ( 1 <<  3 )  # EFT_CHILD_FLAG_SCALE_INHERIT
    InheritRot              = ( 1 <<  4 )  # EFT_CHILD_FLAG_ROTATE_INHERIT
    InheritVel              = ( 1 <<  5 )  # EFT_CHILD_FLAG_VEL_INHERIT
    EmitterFollow           = ( 1 <<  6 )  # EFT_CHILD_FLAG_EMITTER_FOLLOW
    DispParent              = ( 1 <<  7 )  # EFT_CHILD_FLAG_DISP_PARENT
    WorldField              = ( 1 <<  8 )  # EFT_CHILD_FLAG_WORLD_FIELD
    IsPolygon               = ( 1 <<  9 )  # EFT_CHILD_FLAG_IS_POLYGON
    IsEmitterBillboardMtx   = ( 1 << 10 )  # EFT_CHILD_FLAG_IS_EMITTER_BILLBOARD_MTX
    ParentField             = ( 1 << 11 )  # EFT_CHILD_FLAG_PARENT_FIELD
    PreChildDraw            = ( 1 << 12 )  # EFT_CHILD_FLAG_PRE_CHILD_DRAW
    IsTexPatAnim            = ( 1 << 13 )  # EFT_CHILD_FLAG_IS_TEXTURE_PAT_ANIM
    IsTexPatAnimRand        = ( 1 << 14 )  # EFT_CHILD_FLAG_IS_TEXTURE_PAT_ANIM_RAND
    InheritColor1           = ( 1 << 15 )  # EFT_CHILD_FLAG_COLOR1_INHERIT
    InheritColorScale       = ( 1 << 16 )  # EFT_CHILD_FLAG_COLOR_SCALE_INHERIT
    TextureColorOne         = ( 1 << 17 )  # EFT_CHILD_FLAG_TEXTURE_COLOR_ONE
    PrimitiveColorOne       = ( 1 << 18 )  # EFT_CHILD_FLAG_PRIMITIVE_COLOR_ONE
    TextureAlphaOne         = ( 1 << 19 )  # EFT_CHILD_FLAG_TEXTURE_ALPHA_ONE
    PrimitiveAlphaOne       = ( 1 << 20 )  # EFT_CHILD_FLAG_PRIMITIVE_ALPHA_ONE


class FluctuationFlg(enum.IntFlag):
    Enable      = ( 1 << 0 )  # EFT_FLUCTUATION_FALG_ENABLE
    ApplyAlpha  = ( 1 << 1 )  # EFT_FLUCTUATION_FALG_APPLY_ALPHA
    ApplyScale  = ( 1 << 2 )  # EFT_FLUCTUATION_FALG_APPLY_SCLAE


class BillboardType(enum.IntEnum):
    Billboard           = 0  # EFT_BILLBOARD_TYPE_BILLBOARD
    PolygonXY           = 1  # EFT_BILLBOARD_TYPE_POLYGON_XY
    PolygonXZ           = 2  # EFT_BILLBOARD_TYPE_POLYGON_XZ
    VelLook             = 3  # EFT_BILLBOARD_TYPE_VEL_LOOK
    VelLookPolygon      = 4  # EFT_BILLBOARD_TYPE_VEL_LOOK_POLYGON
    HistoricalStripe    = 5  # EFT_BILLBOARD_TYPE_STRIPE
    ConsolidatedStripe  = 6  # EFT_BILLBOARD_TYPE_COMPLEX_STRIPE
    Primitive           = 7  # EFT_BILLBOARD_TYPE_PRIMITIVE (It's not possible to use this type)
    YBillboard          = 8  # EFT_BILLBOARD_TYPE_Y_BILLBOARD


class FieldType(enum.IntEnum):
    Random      = 0  # EFT_FIELD_TYPE_RANDOM
    Magnet      = 1  # EFT_FIELD_TYPE_MAGNET
    Spin        = 2  # EFT_FIELD_TYPE_SPIN
    Collision   = 3  # EFT_FIELD_TYPE_COLLISION
    Convergence = 4  # EFT_FIELD_TYPE_CONVERGENCE
    PosAdd      = 5  # EFT_FIELD_TYPE_POSADD


class FieldMask(enum.IntFlag):
    Random      = ( 1 << FieldType.Random      )  # EFT_FIELD_MASK_RANDOM
    Magnet      = ( 1 << FieldType.Magnet      )  # EFT_FIELD_MASK_MAGNET
    Spin        = ( 1 << FieldType.Spin        )  # EFT_FIELD_MASK_SPIN
    Collision   = ( 1 << FieldType.Collision   )  # EFT_FIELD_MASK_COLLISION
    Convergence = ( 1 << FieldType.Convergence )  # EFT_FIELD_MASK_CONVERGENCE
    PosAdd      = ( 1 << FieldType.PosAdd      )  # EFT_FIELD_MASK_POSADD


class UserDataCallBackID(enum.IntEnum):
    Null    = -1  # EFT_USER_DATA_CALLBACK_ID_NONE
    User_0  =  0  # EFT_USER_DATA_CALLBACK_ID_0
    User_1  =  1  # EFT_USER_DATA_CALLBACK_ID_1
    User_2  =  2  # EFT_USER_DATA_CALLBACK_ID_2
    User_3  =  3  # EFT_USER_DATA_CALLBACK_ID_3
    User_4  =  4  # EFT_USER_DATA_CALLBACK_ID_4
    User_5  =  5  # EFT_USER_DATA_CALLBACK_ID_5
    User_6  =  6  # EFT_USER_DATA_CALLBACK_ID_6
    User_7  =  7  # EFT_USER_DATA_CALLBACK_ID_7


class TextureWrapMode(enum.IntEnum):
    Mirror      = 0  # EFT_TEXTURE_WRAP_TYPE_MIRROR
    Repeat      = 1  # EFT_TEXTURE_WRAP_TYPE_REPEAT
    Clamp       = 2  # EFT_TEXTURE_WRAP_TYPE_CLAMP
  # MirrorOnce  = 3  # EFT_TEXTURE_WRAP_TYPE_MIROOR_ONCE (It's not possible to use this type)

    @staticmethod
    def unpack(wrapMode: int) -> Tuple[Self, Self]:
        return (
            TextureWrapMode(wrapMode & 0xF),
            TextureWrapMode((wrapMode >> 4) & 0xF)
        )

    @staticmethod
    def pack(wrapModeU: Self, wrapModeV: Self) -> int:
        return wrapModeU | wrapModeV << 4


class TextureFilterMode(enum.IntEnum):
    Linear  = 0  # EFT_TEXTURE_FILTER_TYPE_LINEAR
    Near    = 1  # EFT_TEXTURE_FILTER_TYPE_NEAR


class AnimKeyFrameApplyType(enum.IntEnum):
    EmitterRate         =  0  # EFT_ANIM_EM_RATE
    Life                =  1  # EFT_ANIM_LIFE
    EmitterScale_x      =  2  # EFT_ANIM_EM_SX
    EmitterScale_y      =  3  # EFT_ANIM_EM_SY
    EmitterScale_z      =  4  # EFT_ANIM_EM_SZ
    EmitterRot_x        =  5  # EFT_ANIM_EM_RX
    EmitterRot_y        =  6  # EFT_ANIM_EM_RY
    EmitterRot_z        =  7  # EFT_ANIM_EM_RZ
    EmitterTrans_x      =  8  # EFT_ANIM_EM_TX
    EmitterTrans_y      =  9  # EFT_ANIM_EM_TY
    EmitterTrans_z      = 10  # EFT_ANIM_EM_TZ
    GlobalColor0_r      = 11  # EFT_ANIM_COLOR0_R
    GlobalColor0_g      = 12  # EFT_ANIM_COLOR0_G
    GlobalColor0_b      = 13  # EFT_ANIM_COLOR0_B
    GlobalAlpha         = 14  # EFT_ANIM_ALPHA
    FigureVel           = 15  # EFT_ANIM_ALL_DIR_VEL
    DirVel              = 16  # EFT_ANIM_DIR_VEL
    ParticleScale_x     = 17  # EFT_ANIM_PTCL_SX
    ParticleScale_y     = 18  # EFT_ANIM_PTCL_SY
    GlobalColor1_r      = 19  # EFT_ANIM_COLOR1_R
    GlobalColor1_g      = 20  # EFT_ANIM_COLOR1_G
    GlobalColor1_b      = 21  # EFT_ANIM_COLOR1_B
    EmitterFormScale_x  = 22  # EFT_ANIM_EM_FORM_SX
    EmitterFormScale_y  = 23  # EFT_ANIM_EM_FORM_SY
    EmitterFormScale_z  = 24  # EFT_ANIM_EM_FORM_SZ


class PtclRotType(enum.IntEnum):
    NoWork  = 0  # EFT_ROT_TYPE_NO_WORK
    RotX    = 1  # EFT_ROT_TYPE_ROTX
    RotY    = 2  # EFT_ROT_TYPE_ROTY
    RotZ    = 3  # EFT_ROT_TYPE_ROTZ
    RotXYZ  = 4  # EFT_ROT_TYPE_ROTXYZ


class PtclFollowType(enum.IntEnum):
    All     = 0  # EFT_FOLLOW_TYPE_ALL
    Null    = 1  # EFT_FOLLOW_TYPE_NONE
    PosOnly = 2  # EFT_FOLLOW_TYPE_POS_ONLY


class ColorCombinerType(enum.IntEnum):
    Color               = 0  # EFT_COMBINER_TYPE_COLOR
    Texture             = 1  # EFT_COMBINER_TYPE_TEXTURE
    TextureInterpolate  = 2  # EFT_COMBINER_TYPE_TEXTURE_INTERPOLATE
    TextureAdd          = 3  # EFT_COMBINER_TYPE_TEXTURE_ADD


class AlphaBaseCombinerType(enum.IntEnum):
    Mod = 0
    Sub = 1


class AlphaCommonSource(enum.IntEnum):
    Alpha   = 0
    Red     = 1


class AlphaCombinerType(enum.IntEnum):
    Mod  = AlphaBaseCombinerType.Mod | AlphaCommonSource.Alpha << 1  # EFT_ALPHA_COMBINER_TYPE_MOD
    Sub  = AlphaBaseCombinerType.Sub | AlphaCommonSource.Alpha << 1  # EFT_ALPHA_COMBINER_TYPE_SUB
    ModR = AlphaBaseCombinerType.Mod | AlphaCommonSource.Red   << 1  # EFT_ALPHA_COMBINER_TYPE_MOD_R
    SubR = AlphaBaseCombinerType.Sub | AlphaCommonSource.Red   << 1  # EFT_ALPHA_COMBINER_TYPE_SUB_R

    def deconstruct(self) -> Tuple[AlphaBaseCombinerType, AlphaCommonSource]:
        return (
            AlphaBaseCombinerType(self & 1),
            AlphaCommonSource(self >> 1 & 1)
        )

    @staticmethod
    def construct(alphaBaseCombinerType: AlphaBaseCombinerType, alphaCommonSource: AlphaCommonSource) -> Self:
        return AlphaCombinerType(alphaBaseCombinerType | alphaCommonSource << 1)


class DisplaySideType(enum.IntEnum):
    Both    = 0  # EFT_DISPLAYSIDETYPE_BOTH
    Front   = 1  # EFT_DISPLAYSIDETYPE_FRONT
    Back    = 2  # EFT_DISPLAYSIDETYPE_BACK


class BlendType(enum.IntEnum):
    Normal  = 0  # EFT_BLEND_TYPE_NORMAL
    Add     = 1  # EFT_BLEND_TYPE_ADD
    Sub     = 2  # EFT_BLEND_TYPE_SUB
    Screen  = 3  # EFT_BLEND_TYPE_SCREEN
    Mult    = 4  # EFT_BLEND_TYPE_MULT


class ZBufATestType(enum.IntEnum):
    Normal  = 0  # EFT_ZBUFF_ATEST_TYPE_NORMAL
    ZIgnore = 1  # EFT_ZBUFF_ATEST_TYPE_ZIGNORE
    Entity  = 2  # EFT_ZBUFF_ATEST_TYPE_ENTITY


class VolumeType(enum.IntEnum):
    Point               =  0  # EFT_VOLUME_TYPE_POINT
    Circle              =  1  # EFT_VOLUME_TYPE_CIRCLE
    CircleSameDivide    =  2  # EFT_VOLUME_TYPE_CIRCLE_SAME_DIVIDE
    FillCircle          =  3  # EFT_VOLUME_TYPE_CIRCLE_FILL
    Sphere              =  4  # EFT_VOLUME_TYPE_SPHERE
    SphereSameDivide    =  5  # EFT_VOLUME_TYPE_SPHERE_SAME_DIVIDE
    SphereSameDivide64  =  6  # EFT_VOLUME_TYPE_SPHERE_SAME_DIVIDE64
    FillSphere          =  7  # EFT_VOLUME_TYPE_SPHERE_FILL
    Cylinder            =  8  # EFT_VOLUME_TYPE_CYLINDER
    FillCylinder        =  9  # EFT_VOLUME_TYPE_CYLINDER_FILL
    Box                 = 10  # EFT_VOLUME_TYPE_BOX
    FillBox             = 11  # EFT_VOLUME_TYPE_BOX_FILL
    Line                = 12  # EFT_VOLUME_TYPE_LINE
    LineSameDivide      = 13  # EFT_VOLUME_TYPE_LINE_SAME_DIVIDE
    Rectangle           = 14  # EFT_VOLUME_TYPE_RECTANGLE


class EmitterFlg(enum.IntFlag):
    EnableSortParticle      = ( 1 <<  9 )  # EFT_EMITTER_FLAG_ENABLE_SORTPARTICLE
    ReverseOrderParticle    = ( 1 << 10 )  # EFT_EMITTER_FLAG_REVERSE_ORDER_PARTICLE
    Texture0ColorOne        = ( 1 << 11 )  # EFT_EMITTER_FLAG_TEXTURE0_COLOR_ONE
    Texture1ColorOne        = ( 1 << 12 )  # EFT_EMITTER_FLAG_TEXTURE1_COLOR_ONE
    PrimitiveColorOne       = ( 1 << 13 )  # EFT_EMITTER_FLAG_PRIMITIVE_COLOR_ONE
    Texture0AlphaOne        = ( 1 << 14 )  # EFT_EMITTER_FLAG_TEXTURE0_ALPHA_ONE
    Texture1AlphaOne        = ( 1 << 15 )  # EFT_EMITTER_FLAG_TEXTURE1_ALPHA_ONE
    PrimitiveAlphaOne       = ( 1 << 16 )  # EFT_EMITTER_FLAG_PRIMITIVE_ALPHA_ONE


class MeshType(enum.IntEnum):
    Particle    = 0  # EFT_MESH_TYPE_PARTICLE
    Primitive   = 1  # EFT_MESH_TYPE_PRIMITIVE
    Stripe      = 2  # EFT_MESH_TYPE_STRIPE


class VertexRotationVariation(enum.IntEnum):
    NoUse   = 0  # EFT_VERTEX_SHADER_ROTATION_VARIATION_NO_USE
    Use     = 1  # EFT_VERTEX_SHADER_ROTATION_VARIATION_USE


class StripeFlg(enum.IntFlag):
    EmitterCoord    = ( 1 << 0 )  # EFT_STRIPE_FLAG_EMITTER_COORD


class StripeType(enum.IntEnum):
    Billboard       = 0  # EFT_STRIPE_TYPE_BILLBOARD
    EmitterMatrix   = 1  # EFT_STRIPE_TYPE_EMITTER_MATRIX
    EmitterUpDown   = 2  # EFT_STRIPE_TYPE_EMITTER_UP_DOWN
    Max             = 3  # EFT_STRIPE_TYPE_MAX


class StripeOption(enum.IntEnum):
    Normal  = 0  # EFT_STRIPE_OPTION_TYPE_NORMAL
    Cross   = 1  # EFT_STRIPE_OPTION_TYPE_CROSS


class StripeConnectOption(enum.IntEnum):
    Normal  = 0  # EFT_STRIPE_CONNECT_OPTION_NORMAL
    Head    = 1  # EFT_STRIPE_CONNECT_OPTION_HEAD
    Emitter = 2  # EFT_STRIPE_CONNECT_OPTION_EMITTER


class StripeTexCoordOption(enum.IntEnum):
    Full        = 0  # EFT_STRIPE_TEXCOORD_OPTION_TYPE_FULL
    Division    = 1  # EFT_STRIPE_TEXCOORD_OPTION_TYPE_DIVISION


class FragmentShaderVariation(enum.IntEnum):
    Normal      = 0  # EFT_FRAGMENT_SHADER_TYPE_VARIATION_PARTICLE
    Refraction  = 1  # EFT_FRAGMENT_SHADER_TYPE_VARIATION_REFRACT_PARTICLE
    Distortion  = 2  # EFT_FRAGMENT_SHADER_TYPE_VARIATION_DISTORTION_PARTICLE


class FragmentTextureVariation(enum.IntEnum):
    First   = 0  # EFT_FRAGMENT_SHADER_TEXTURE_VARIATION_0
    Second  = 1  # EFT_FRAGMENT_SHADER_TEXTURE_VARIATION_1


class ColorAlphaBlendType(enum.IntEnum):
    Mod = 0  # EFT_COLOR_BLEND_TYPE_MOD
    Add = 1  # EFT_COLOR_BLEND_TYPE_ADD
    Sub = 2  # EFT_COLOR_BLEND_TYPE_SUB


class AnimKeyFrameInterpolationType(enum.IntEnum):
    Linear  = 0  # EFT_ANIM_KEY_FRAME_LINEAR
    Smooth  = 1  # EFT_ANIM_KEY_FRAME_SMOOTH


class TextureAddressing(enum.IntEnum):
    U1V1    = 0  # EFT_TEX_ADDRESSING_NORMAL
    U2V1    = 1  # EFT_TEX_ADDRESSING_MIRROR_U2
    U1V2    = 2  # EFT_TEX_ADDRESSING_MIRROR_V2
    U2V2    = 3  # EFT_TEX_ADDRESSING_MIRROR_U2_V2


class UvShiftAnimMode(enum.IntEnum):
    Null    = 0  # EFT_UV_SHIFT_ANIM_NONE
    Trans   = 1  # EFT_UV_SHIFT_ANIM_SCROLL
    Scale   = 2  # EFT_UV_SHIFT_ANIM_SCALE
    Rot     = 3  # EFT_UV_SHIFT_ANIM_ROT
    All     = 4  # EFT_UV_SHIFT_ANIM_ALL


class FieldMagnetFlg(enum.IntFlag):
    TargetX = ( 1 << 0 )  # EFT_MAGNET_FLAG_X
    TargetY = ( 1 << 1 )  # EFT_MAGNET_FLAG_Y
    TargetZ = ( 1 << 2 )  # EFT_MAGNET_FLAG_Z


class FieldCollisionReaction(enum.IntEnum):
    Cessation   = 0  # EFT_FIELD_COLLISION_REACTION_CESSER
    Reflection  = 1  # EFT_FIELD_COLLISION_REACTION_REFLECTION


class ParticleDrawOrder(enum.Enum):
    Ascending   = enum.auto()
    Descending  = enum.auto()
    ZSort       = enum.auto()


class ColorSource(enum.IntEnum):
    RGB = 0
    One = 1


class AlphaSource(enum.IntEnum):
    Pass    = 0
    One     = 1


class EmissionIntervalType(enum.IntEnum):
    Time        = 0
    Distance    = 1


class ArcOpeningType(enum.IntEnum):
    Longitude   = 0
    Latitude    = 1


class TexturePatternAnimMode(enum.Enum):
    Null    = enum.auto()
    LifeFit = enum.auto()
    Loop    = enum.auto()
    Random  = enum.auto()
    Clamp   = enum.auto()


class FieldSpinAxis(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2

    x = X
    y = Y
    z = Z


class VolumeLatitudeDir(enum.Enum):
    PosX = ( 1.0,  0.0,  0.0)
    NegX = (-1.0,  0.0,  0.0)
    PosY = ( 0.0,  1.0,  0.0)
    NegY = ( 0.0, -1.0,  0.0)
    PosZ = ( 0.0,  0.0,  1.0)
    NegZ = ( 0.0,  0.0, -1.0)

    x = X = PosX
    y = Y = PosY
    z = Z = PosZ


LoadTextureCache: List[str] = []
LoadPrimitiveCache: List[str] = []


def ResolveTextureIndex(filename: Any, originalDataFormat: Any) -> Tuple[int, int, TextureFormat]:
    if filename is None:
        return -1, -1, TextureFormat.Invalid
    assert isinstance(filename, str)
    isGTX = filename.endswith('.gtx')
    isPNG = filename.endswith('.png')
    assert isGTX or isPNG
    if originalDataFormat is None:
        ret_originalDataFormat = TextureFormat.Unorm_RGBA8
    else:
        assert originalDataFormat == 'RGB' or originalDataFormat == 'RGBA'
        ret_originalDataFormat = TextureFormat.Unorm_RGBA8 if originalDataFormat == 'RGBA' else TextureFormat.Unorm_RGB8
    try:
        ret_index = LoadTextureCache.index(filename)
    except ValueError:
        LoadTextureCache.append(filename)
        ret_index = len(LoadTextureCache) - 1
    if isPNG:
        return ret_index, -1, ret_originalDataFormat
    else:
        return -1, ret_index, ret_originalDataFormat


def ResolvePrimitiveIndex(filename: Any) -> int:
    if filename is None:
        return 0xFFFFFFFF
    assert isinstance(filename, str)
    try:
        return LoadPrimitiveCache.index(filename)
    except ValueError:
        LoadPrimitiveCache.append(filename)
        return len(LoadPrimitiveCache) - 1


class nw__eft__HeaderData:
    structSize = struct.calcsize('>4sI11i') + struct.calcsize('<3i')
    assert structSize == 0x40

    numEmitterSet: int
    namePos: int
    nameTblPos: int
    textureTblPos: int
    textureTblSize: int
    shaderTblPos: int
    shaderTblSize: int
    animKeyTblPos: int
    animKeyTblSize: int
    primitiveTblPos: int
    primitiveTblSize: int
    totalShaderSize: int
    totalEmitterSize: int

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            magic,                  # Magic code ('SPBD': SEAD Ptcl Binary Data)
            version,                # Version
            self.numEmitterSet,     # Number of emitter sets
            self.namePos,           # Name (offset from beginning of name table)
            self.nameTblPos,        # Name table (offset from beginning of file)
            self.textureTblPos,     # Texture table (offset from beginning of file)
            self.textureTblSize,    # Texture table size
            self.shaderTblPos,      # Shader table (offset from beginning of file)
            self.shaderTblSize,     # Shader table size
            self.animKeyTblPos,     # Keyframe animation table (offset from beginning of file)
            self.animKeyTblSize,    # Keyframe animation table size
            self.primitiveTblPos,   # Primitive table (offset from beginning of file)
            self.primitiveTblSize,  # Primitive table size
        ) = struct.unpack_from('>4sI11i', data, pos); pos += struct.calcsize('>4sI11i')

        assert magic == b'SPBD'
        assert version == SystemConstants.BinaryVersion

        (
            totalTextureSize,       # Total texture size        (in little endian!!!)
            self.totalShaderSize,   # Total shader binary size  (in little endian!!!)
            self.totalEmitterSize   # Total emitter binary size (in little endian!!!)
        ) = struct.unpack_from('<3i', data, pos)

        assert totalTextureSize == self.textureTblSize

    def save(self) -> bytes:
        return b''.join((
            struct.pack(
                '>4sI11i',
                b'SPBD',
                SystemConstants.BinaryVersion,
                self.numEmitterSet,
                self.namePos,
                self.nameTblPos,
                self.textureTblPos,
                self.textureTblSize,
                self.shaderTblPos,
                self.shaderTblSize,
                self.animKeyTblPos,
                self.animKeyTblSize,
                self.primitiveTblPos,
                self.primitiveTblSize
            ),
            struct.pack(
                '<3i',
                self.textureTblSize,
                self.totalShaderSize,
                self.totalEmitterSize
            )
        ))


class nw__eft__ShaderImageInformation:
    structSize = struct.calcsize('>4I')
    assert structSize == 0x10

    shaderNum: int
    totalSize: int
    offsetShaderBinInfo: int

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            self.shaderNum,             # Shader count
            self.totalSize,             # Shader total size
            offsetShaderSrcInfo,        # ShaderSrcInformation entries (offset from beginning of this structure)
            self.offsetShaderBinInfo    # ShaderInformation entries (offset from beginning of this structure)
        ) = struct.unpack_from('>4I', data, pos)

        assert offsetShaderSrcInfo == nw__eft__ShaderImageInformation.structSize

    def save(self) -> bytes:
        return struct.pack(
            '>4I',
            self.shaderNum,
            self.totalSize,
            nw__eft__ShaderImageInformation.structSize,
            self.offsetShaderBinInfo
        )


class nw__eft__ShaderSrcInformation:
    class sourceCodeTable:
        structSize = struct.calcsize('>2I')
        assert structSize == 8

        size: int
        offset: int

        def load(self, data: ByteString, pos: int = 0) -> None:
            (
                self.size,  # Shader source code size
                self.offset # Offset
            ) = struct.unpack_from('>2I', data, pos)

        def save(self) -> bytes:
            return struct.pack(
                '>2I',
                self.size,
                self.offset
            )

    structSize = struct.calcsize('>2I') + 8 * sourceCodeTable.structSize
    assert structSize == 0x48

    shaderSourceNum: int
    shaderSourceTotalSize: int
    vshParticle: sourceCodeTable
    fshParticle: sourceCodeTable
    vshStripe: sourceCodeTable
  # gshStripe: sourceCodeTable
    vshUser: sourceCodeTable
    fshUser: sourceCodeTable
    vshParticleDeclaration: sourceCodeTable
    fshParticleDeclaration: sourceCodeTable

    def __init__(self) -> None:
        self.vshParticle = nw__eft__ShaderSrcInformation.sourceCodeTable()
        self.fshParticle = nw__eft__ShaderSrcInformation.sourceCodeTable()
        self.vshStripe = nw__eft__ShaderSrcInformation.sourceCodeTable()
        self.vshUser = nw__eft__ShaderSrcInformation.sourceCodeTable()
        self.fshUser = nw__eft__ShaderSrcInformation.sourceCodeTable()
        self.vshParticleDeclaration = nw__eft__ShaderSrcInformation.sourceCodeTable()
        self.fshParticleDeclaration = nw__eft__ShaderSrcInformation.sourceCodeTable()

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            self.shaderSourceNum,       # Shader source code count
            self.shaderSourceTotalSize  # Shader source code total size
        ) = struct.unpack_from('>2I', data, pos); pos += struct.calcsize('>2I')

        gshStripe = nw__eft__ShaderSrcInformation.sourceCodeTable()

        self.vshParticle.load(data, pos); pos += nw__eft__ShaderSrcInformation.sourceCodeTable.structSize             # Particle              shader (vertex)
        self.fshParticle.load(data, pos); pos += nw__eft__ShaderSrcInformation.sourceCodeTable.structSize             # Particle              shader (fragment)
        self.vshStripe.load(data, pos); pos += nw__eft__ShaderSrcInformation.sourceCodeTable.structSize               # Stripe                shader (vertex)
        gshStripe.load(data, pos); pos += nw__eft__ShaderSrcInformation.sourceCodeTable.structSize                    # Stripe                shader (geometry)
        self.vshUser.load(data, pos); pos += nw__eft__ShaderSrcInformation.sourceCodeTable.structSize                 # User                  shader (vertex)
        self.fshUser.load(data, pos); pos += nw__eft__ShaderSrcInformation.sourceCodeTable.structSize                 # User                  shader (fragment)
        self.vshParticleDeclaration.load(data, pos); pos += nw__eft__ShaderSrcInformation.sourceCodeTable.structSize  # Particle Declaration  shader (vertex)
        self.fshParticleDeclaration.load(data, pos)                                                             # Particle Declaration  shader (fragment)

        assert gshStripe.size == 0 and gshStripe.offset == 0

    def save(self) -> bytes:
        gshStripe = nw__eft__ShaderSrcInformation.sourceCodeTable()
        gshStripe.size = 0
        gshStripe.offset = 0

        return b''.join((
            struct.pack(
                '>2I',
                self.shaderSourceNum,
                self.shaderSourceTotalSize
            ),
            self.vshParticle.save(),
            self.fshParticle.save(),
            self.vshStripe.save(),
            gshStripe.save(),
            self.vshUser.save(),
            self.fshUser.save(),
            self.vshParticleDeclaration.save(),
            self.fshParticleDeclaration.save()
        ))


class nw__eft__PrimitiveImageInformation:
    structSize = struct.calcsize('>3I')
    assert structSize == 0xC

    primitiveNum: int
    totalSize: int

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            self.primitiveNum,
            self.totalSize,
            offsetPrimitiveTableInfo
        ) = struct.unpack_from('>3I', data, pos)

        assert offsetPrimitiveTableInfo == nw__eft__PrimitiveImageInformation.structSize

    def save(self) -> bytes:
        return struct.pack(
            '>3I',
            self.primitiveNum,
            self.totalSize,
            nw__eft__PrimitiveImageInformation.structSize
        )


class nw__eft__PrimitiveTableInfo:
    class PrimDataTable:
        structSize = struct.calcsize('>4I')
        assert structSize == 0x10

        column: int
        offset: int
        size: int

        def load(self, data: ByteString, pos: int = 0) -> None:
            (
                count,          # Total number of float's
                self.column,    # Number of float's per entry (i.e., Vec2 -> 2, Vec3 -> 3, Vec4 -> 4)
                self.offset,    # Offset to attribute data
                self.size       # Size of attribute data
            ) = struct.unpack_from('>4I', data, pos)

            assert count * F32_SIZE == self.size

        def save(self) -> bytes:
            assert self.size % F32_SIZE == 0

            return struct.pack(
                '>4I',
                self.size // F32_SIZE,
                self.column,
                self.offset,
                self.size
            )

    structSize = 5 * PrimDataTable.structSize
    assert structSize == 0x50

    pos: PrimDataTable
    normal: PrimDataTable
    color: PrimDataTable
    texCoord: PrimDataTable
    index: PrimDataTable

    def __init__(self) -> None:
        self.pos = nw__eft__PrimitiveTableInfo.PrimDataTable()
        self.normal = nw__eft__PrimitiveTableInfo.PrimDataTable()
        self.color = nw__eft__PrimitiveTableInfo.PrimDataTable()
        self.texCoord = nw__eft__PrimitiveTableInfo.PrimDataTable()
        self.index = nw__eft__PrimitiveTableInfo.PrimDataTable()

    def load(self, data: ByteString, pos: int = 0) -> None:
        self.pos.load(data, pos); pos += nw__eft__PrimitiveTableInfo.PrimDataTable.structSize
        self.normal.load(data, pos); pos += nw__eft__PrimitiveTableInfo.PrimDataTable.structSize
        self.color.load(data, pos); pos += nw__eft__PrimitiveTableInfo.PrimDataTable.structSize
        self.texCoord.load(data, pos); pos += nw__eft__PrimitiveTableInfo.PrimDataTable.structSize
        self.index.load(data, pos)

    def save(self) -> bytes:
        return b''.join((
            self.pos.save(),
            self.normal.save(),
            self.color.save(),
            self.texCoord.save(),
            self.index.save()
        ))


class nw__eft__Primitive:
    pos: Tuple[Vec3Numeric, ...]
    nor: Optional[Tuple[Vec3Numeric, ...]]
    col: Optional[Tuple[Color4Numeric, ...]]
    tex: Tuple[Vec2Numeric, ...]
    idx: Tuple[int, ...]

    def Initialize(
        self, binData: ByteString,
        posSrcPos: int, posSrcSize: int, posSrcCol: int,
        norSrcPos: int, norSrcSize: int, norSrcCol: int,
        colSrcPos: int, colSrcSize: int, colSrcCol: int,
        texSrcPos: int, texSrcSize: int, texSrcCol: int,
        idxSrcPos: int, idxSrcSize: int, idxSrcCol: int
    ) -> None:
        assert posSrcPos >= 0 and texSrcPos >= 0 and idxSrcPos >= 0

        assert posSrcCol == 3  # Vec3
        assert posSrcSize % VEC3_SIZE == 0
        posSrcNum = posSrcSize // VEC3_SIZE
        self.pos = tuple(struct.unpack_from(VEC3_FMT, binData, posSrcPos + i * VEC3_SIZE) for i in range(posSrcNum))

        assert norSrcCol == 3
        if norSrcPos >= 0:
            assert norSrcCol == 3  # Vec3
            assert norSrcSize % VEC3_SIZE == 0
            norSrcNum = norSrcSize // VEC3_SIZE
            assert norSrcNum == posSrcNum
            self.nor = tuple(struct.unpack_from(VEC3_FMT, binData, norSrcPos + i * VEC3_SIZE) for i in range(norSrcNum))
        else:
            assert norSrcCol == 0
            self.nor = None

        if colSrcPos >= 0:
            assert colSrcCol == 4  # Vec4
            assert colSrcSize % VEC4_SIZE == 0
            colSrcNum = colSrcSize // VEC4_SIZE
            assert colSrcNum == posSrcNum
            self.col = tuple(struct.unpack_from(VEC4_FMT, binData, colSrcPos + i * VEC4_SIZE) for i in range(colSrcNum))
        else:
            assert colSrcCol == 0
            self.col = None

        assert texSrcCol == 2  # Vec2
      # assert texSrcSize != 0
        assert texSrcSize % VEC2_SIZE == 0
        texSrcNum = texSrcSize // VEC2_SIZE
        assert texSrcNum == posSrcNum
        self.tex = tuple(struct.unpack_from(VEC2_FMT, binData, texSrcPos + i * VEC2_SIZE) for i in range(texSrcNum))

        assert idxSrcCol == 3  # WHAT???
        assert idxSrcSize != 0
        assert idxSrcSize % U32_SIZE == 0
        indexNum = idxSrcSize // U32_SIZE
        self.idx = tuple(struct.unpack_from(U32_FMT, binData, idxSrcPos + i * U32_SIZE)[0] for i in range(indexNum))
        assert max(self.idx) < posSrcNum

    def toGLB(self, file_path: str) -> None:
        indexNum = len(self.idx)
        vertexNum = len(self.pos)
        if self.nor is not None:
            assert len(self.nor) == vertexNum
        if self.col is not None:
            assert len(self.col) == vertexNum
        assert len(self.tex) == vertexNum
        assert max(self.idx) < vertexNum

        bufferIndex = 0
        accessors: List[pygltflib.Accessor] = []
        bufferViews: List[pygltflib.BufferView] = []
        buffer = bytearray()

        indexBufferIndex = bufferIndex; bufferIndex += 1
        accessors.append(
            pygltflib.Accessor(
                bufferView=indexBufferIndex,
                componentType=pygltflib.UNSIGNED_INT,
                count=indexNum,
                type=pygltflib.SCALAR
            )
        )
        bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(buffer),
                byteLength=U32_SIZE * indexNum,
                target=pygltflib.ELEMENT_ARRAY_BUFFER
            )
        )
        for index in self.idx:
            buffer += struct.pack(U32_FMT_LE, index)

        positionBufferIndex = bufferIndex; bufferIndex += 1
        accessors.append(
            pygltflib.Accessor(
                bufferView=positionBufferIndex,
                componentType=pygltflib.FLOAT,
                count=vertexNum,
                type=pygltflib.VEC3,
                max=(
                    max(self.pos, key=lambda position: position[0])[0],
                    max(self.pos, key=lambda position: position[1])[1],
                    max(self.pos, key=lambda position: position[2])[2]
                ),
                min=(
                    min(self.pos, key=lambda position: position[0])[0],
                    min(self.pos, key=lambda position: position[1])[1],
                    min(self.pos, key=lambda position: position[2])[2]
                )
            )
        )
        bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(buffer),
                byteLength=VEC3_SIZE * vertexNum,
                target=pygltflib.ARRAY_BUFFER
            )
        )
        for position in self.pos:
            buffer += struct.pack(VEC3_FMT_LE, *position)

        if self.nor is None:
            normalBufferIndex = None
        else:
            normalBufferIndex = bufferIndex; bufferIndex += 1
            accessors.append(
                pygltflib.Accessor(
                    bufferView=normalBufferIndex,
                    componentType=pygltflib.FLOAT,
                    count=vertexNum,
                    type=pygltflib.VEC3
                )
            )
            bufferViews.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(buffer),
                    byteLength=VEC3_SIZE * vertexNum,
                    target=pygltflib.ARRAY_BUFFER
                )
            )
            for normal in self.nor:
                buffer += struct.pack(VEC3_FMT_LE, *normal)

        if self.col is None:
            colorBufferIndex = None
        else:
            colorBufferIndex = bufferIndex; bufferIndex += 1
            accessors.append(
                pygltflib.Accessor(
                    bufferView=colorBufferIndex,
                    componentType=pygltflib.FLOAT,
                    count=vertexNum,
                    type=pygltflib.VEC4
                )
            )
            bufferViews.append(
                pygltflib.BufferView(
                    buffer=0,
                    byteOffset=len(buffer),
                    byteLength=VEC4_SIZE * vertexNum,
                    target=pygltflib.ARRAY_BUFFER
                )
            )
            for color in self.col:
                buffer += struct.pack(VEC4_FMT_LE, *color)

        texCoordBufferIndex = bufferIndex; bufferIndex += 1
        accessors.append(
            pygltflib.Accessor(
                bufferView=texCoordBufferIndex,
                componentType=pygltflib.FLOAT,
                count=vertexNum,
                type=pygltflib.VEC2
            )
        )
        bufferViews.append(
            pygltflib.BufferView(
                buffer=0,
                byteOffset=len(buffer),
                byteLength=VEC2_SIZE * vertexNum,
                target=pygltflib.ARRAY_BUFFER
            )
        )
        for texCoord in self.tex:
            buffer += struct.pack(VEC2_FMT_LE, *texCoord)

        glb = pygltflib.GLTF2(
            scene=0,
            scenes=[pygltflib.Scene(nodes=[0])],
            nodes=[pygltflib.Node(mesh=0)],
            meshes=[
                pygltflib.Mesh(
                    primitives=[
                        pygltflib.Primitive(
                            attributes=pygltflib.Attributes(
                                POSITION=positionBufferIndex,
                                NORMAL=normalBufferIndex,
                                COLOR_0=colorBufferIndex,
                                TEXCOORD_0=texCoordBufferIndex
                            ),
                            indices=indexBufferIndex
                        )
                    ]
                )
            ],
            accessors=accessors,
            bufferViews=bufferViews,
            buffers=[
                pygltflib.Buffer(
                    byteLength=len(buffer)
                )
            ]
        )
        glb.set_binary_blob(buffer)
        glb.save(file_path)

    def fromGLB(self, file_path: str) -> None:
        glb = pygltflib.GLTF2().load(file_path)
        assert glb is not None
        buffer: Optional[ByteString] = glb.binary_blob()
        assert buffer is not None
        assert glb.buffers[0].byteLength == len(buffer)

        primitive = glb.meshes[0].primitives[0]
        accessors = glb.accessors
        bufferViews = glb.bufferViews

        indexBufferIndex = primitive.indices
        assert indexBufferIndex is not None and indexBufferIndex >= 0
        indexBufferAccessor: pygltflib.Accessor = accessors[indexBufferIndex]
        assert indexBufferAccessor.byteOffset == 0
        assert indexBufferAccessor.componentType == pygltflib.UNSIGNED_INT
        assert indexBufferAccessor.type == pygltflib.SCALAR
        indexNum = indexBufferAccessor.count
        indexBufferView: pygltflib.BufferView = bufferViews[indexBufferAccessor.bufferView]
        assert indexBufferView.buffer == 0
        assert indexBufferView.byteLength == U32_SIZE * indexNum
        assert indexBufferView.target == pygltflib.ELEMENT_ARRAY_BUFFER
        idxbyteOffset = indexBufferView.byteOffset
        self.idx = tuple(struct.unpack_from(U32_FMT_LE, buffer, idxbyteOffset + i * U32_SIZE)[0] for i in range(indexNum))

        positionBufferIndex = primitive.attributes.POSITION
        assert positionBufferIndex is not None and positionBufferIndex >= 0
        positionBufferAccessor: pygltflib.Accessor = accessors[positionBufferIndex]
        assert positionBufferAccessor.byteOffset == 0
        assert positionBufferAccessor.componentType == pygltflib.FLOAT
        assert positionBufferAccessor.type == pygltflib.VEC3
        vertexNum = positionBufferAccessor.count
        positionBufferView: pygltflib.BufferView = bufferViews[positionBufferAccessor.bufferView]
        assert positionBufferView.buffer == 0
        assert positionBufferView.byteLength == VEC3_SIZE * vertexNum
        assert positionBufferView.target == pygltflib.ARRAY_BUFFER
        posbyteOffset = positionBufferView.byteOffset
        self.pos = tuple(struct.unpack_from(VEC3_FMT_LE, buffer, posbyteOffset + i * VEC3_SIZE) for i in range(vertexNum))

        normalBufferIndex = primitive.attributes.NORMAL
        if not (normalBufferIndex is not None and normalBufferIndex >= 0):
            self.nor = None
        else:
            normalBufferAccessor: pygltflib.Accessor = accessors[normalBufferIndex]
            assert normalBufferAccessor.byteOffset == 0
            assert normalBufferAccessor.componentType == pygltflib.FLOAT
            assert normalBufferAccessor.type == pygltflib.VEC3
            assert normalBufferAccessor.count == vertexNum
            normalBufferView: pygltflib.BufferView = bufferViews[normalBufferAccessor.bufferView]
            assert normalBufferView.buffer == 0
            assert normalBufferView.byteLength == VEC3_SIZE * vertexNum
            assert normalBufferView.target == pygltflib.ARRAY_BUFFER
            norbyteOffset = normalBufferView.byteOffset
            self.nor = tuple(struct.unpack_from(VEC3_FMT_LE, buffer, norbyteOffset + i * VEC3_SIZE) for i in range(vertexNum))

        colorBufferIndex = primitive.attributes.COLOR_0
        if not (colorBufferIndex is not None and colorBufferIndex >= 0):
            self.col = None
        else:
            colorBufferAccessor: pygltflib.Accessor = accessors[colorBufferIndex]
            assert colorBufferAccessor.byteOffset == 0
            assert colorBufferAccessor.componentType == pygltflib.FLOAT
            assert colorBufferAccessor.type == pygltflib.VEC4
            assert colorBufferAccessor.count == vertexNum
            colorBufferView: pygltflib.BufferView = bufferViews[colorBufferAccessor.bufferView]
            assert colorBufferView.buffer == 0
            assert colorBufferView.byteLength == VEC4_SIZE * vertexNum
            assert colorBufferView.target == pygltflib.ARRAY_BUFFER
            colbyteOffset = colorBufferView.byteOffset
            self.col = tuple(struct.unpack_from(VEC4_FMT_LE, buffer, colbyteOffset + i * VEC4_SIZE) for i in range(vertexNum))

        texCoordBufferIndex = primitive.attributes.TEXCOORD_0
        assert texCoordBufferIndex is not None and texCoordBufferIndex >= 0
        texCoordBufferAccessor: pygltflib.Accessor = accessors[texCoordBufferIndex]
        assert texCoordBufferAccessor.byteOffset == 0
        assert texCoordBufferAccessor.componentType == pygltflib.FLOAT
        assert texCoordBufferAccessor.type == pygltflib.VEC2
        assert texCoordBufferAccessor.count == vertexNum
        texCoordBufferView: pygltflib.BufferView = bufferViews[texCoordBufferAccessor.bufferView]
        assert texCoordBufferView.buffer == 0
        assert texCoordBufferView.byteLength == VEC2_SIZE * vertexNum
        assert texCoordBufferView.target == pygltflib.ARRAY_BUFFER
        texbyteOffset = texCoordBufferView.byteOffset
        self.tex = tuple(struct.unpack_from(VEC2_FMT_LE, buffer, texbyteOffset + i * VEC2_SIZE) for i in range(vertexNum))


class nw__eft__EmitterSetData:
    structSize = struct.calcsize('>I4xi4x2i4x')
    assert structSize == 0x1C

    userDataNum1: int
    userDataNum2: int
    userDataBit: int
    namePos: int
    name: str
    numEmitter: int
    emitterTblPos: int
  # emitterTbl: List[nw__eft__EmitterTblData]

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            userData,               # User data
          # self.lastUpdateDate,    # Last modified (never set)
            self.namePos,           # Name (offset from beginning of name table)
          # self.name,              # Name (set at runtime)
            self.numEmitter,        # Emitter count
            self.emitterTblPos,     # Emitter table (offset from the beginning of the file)
          # self.emitterTbl         # Emitter table (set at runtime)
        ) = struct.unpack_from('>I4xi4x2i4x', data, pos)

        # User data (1/3: 1-byte number 1)
        self.userDataNum1 = userData & 0xFF
        # User data (2/3: 1-byte number 2)
        self.userDataNum2 = userData >> 8 & 0xFF
        # User data (3/3: 16-bit bitfield)
        self.userDataBit = userData >> 16

    def save(self) -> bytes:
        assert 0 <= self.userDataNum1 <= 0xFF
        assert 0 <= self.userDataNum2 <= 0xFF
        assert 0 <= self.userDataBit <= 0xFFFF

        return struct.pack(
            '>I4xi4x2i4x',
            (
                self.userDataNum1 |
                self.userDataNum2 << 8 |
                self.userDataBit << 16
            ),
          # self.lastUpdateDate,
            self.namePos,
          # self.name,
            self.numEmitter,
            self.emitterTblPos,
          # self.emitterTbl
        )


class nw__eft__EmitterTblData:
    structSize = struct.calcsize('>i4x')
    assert structSize == 8

    emitterPos: int
  # emitter: nw__eft__EmitterData

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            self.emitterPos,    # Emitter (offset from beginning of file)
          # self.emitter        # Emitter (set at runtime)
        ) = struct.unpack_from('>i4x', data, pos)

    def save(self) -> bytes:
        return struct.pack(
            '>i4x',
            self.emitterPos,
          # self.emitter
        )


class nw__eft__TextureRes:
    structSize = (
        struct.calcsize('>2H4I2B2x2I') +
        struct.calcsize('>13I') +
        struct.calcsize('>2fI2iI2iI156x')
    )
    assert structSize == 0x114

    width: int
    height: int
    tileMode: GX2TileMode
    swizzle: int
    alignment: int
    pitch: int
    wrapModeU: TextureWrapMode
    wrapModeV: TextureWrapMode
    filterMode: TextureFilterMode
    mipLevel: int
    compSel: int
    mipOffset: Tuple[int, int, int, int, int, int, int, int, int, int, int, int, int]
    enableMipLevel: Numeric
    mipMapBias: Numeric
    originalDataFormat: TextureFormat
    originalDataPos: int
    originalDataSize: int
    nativeDataFormat: TextureFormat
    nativeDataSize: int
    nativeDataPos: int
    imageIndex: int
    gx2TextureIndex: int

    def __init__(self) -> None:
        self.width = 0
        self.height = 0
        self.tileMode = GX2TileMode(0)
        self.swizzle = 0
        self.alignment = 0
        self.pitch = 0
        self.wrapModeU = TextureWrapMode.Mirror
        self.wrapModeV = TextureWrapMode.Mirror
        self.filterMode = TextureFilterMode.Linear
        self.mipLevel = 0
        self.compSel = 0
        self.mipOffset = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        self.enableMipLevel = ENABLE_MIP_LEVEL_MAX
        self.mipMapBias = 0.0
        self.originalDataFormat = TextureFormat.Invalid
        self.originalDataPos = 0
        self.originalDataSize = 0
        self.nativeDataFormat = TextureFormat.Invalid
        self.nativeDataSize = 0
        self.nativeDataPos = 0
        self.imageIndex = -1
        self.gx2TextureIndex = -1

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            self.width,     # Texture width
            self.height,    # Texture height
            tileMode,       # GX2Surface::tileMode
            self.swizzle,   # GX2Surface::swizzle
            self.alignment, # GX2Surface::alignment
            self.pitch,     # GX2Surface::pitch
            wrapMode,       # Wrap mode (Lower: U, Higher: V)
            filterMode,     # Filter mode
            self.mipLevel,  # Mip level count
            self.compSel    # Component swap
        ) = struct.unpack_from('>2H4I2B2x2I', data, pos); pos += struct.calcsize('>2H4I2B2x2I')

        self.tileMode = GX2TileMode(tileMode)

        self.wrapModeU, self.wrapModeV = TextureWrapMode.unpack(wrapMode)
        self.filterMode = TextureFilterMode(filterMode)

        # GX2Surface::mipOffset
        self.mipOffset = struct.unpack_from('>13I', data, pos); pos += struct.calcsize('>13I')

        (
            enableMipLevel,         # Mip levels valid values (0.0 to 15.99)
            mipMapBias,        # Mipmap level bias
            originalDataFormat,     # Original texture format
            self.originalDataPos,   # Original texture data (offset from beginning of texture table)
            self.originalDataSize,  # Original texture data size
            nativeDataFormat,       # Native texture format
            self.nativeDataSize,    # Native texture data (offset from beginning of texture table)
            self.nativeDataPos,     # Native texture data size
            handle,
          # self.gx2Texture
        ) = struct.unpack_from('>2fI2iI2iI156x', data, pos)

        self.enableMipLevel = F32StandardToRegular(enableMipLevel, ENABLE_MIP_LEVEL_MAX_STD, ENABLE_MIP_LEVEL_MAX)
        assert 0.0 <= self.enableMipLevel <= ENABLE_MIP_LEVEL_MAX

        self.mipMapBias = F32StandardToRegular(mipMapBias, MIP_MAP_BIAS_MAX_STD, MIP_MAP_BIAS_MAX)
        assert -32.0 <= self.mipMapBias <= MIP_MAP_BIAS_MAX

        self.originalDataFormat = TextureFormat(originalDataFormat)
        self.nativeDataFormat = TextureFormat(nativeDataFormat)

        assert handle == 0

        self.imageIndex = -1        # Texture handle for original data
        self.gx2TextureIndex = -1   # GX2Texture for native data

    def save(self) -> bytes:
        assert 0 <= self.enableMipLevel <= ENABLE_MIP_LEVEL_MAX

        return b''.join((
            struct.pack(
                '>2H4I2B2x2I',
                self.width,
                self.height,
                self.tileMode,
                self.swizzle,
                self.alignment,
                self.pitch,
                TextureWrapMode.pack(self.wrapModeU, self.wrapModeV),
                self.filterMode,
                self.mipLevel,
                self.compSel
            ),
            struct.pack('>13I', *self.mipOffset),
            struct.pack(
                '>2fI2iI2iI156x',
                self.enableMipLevel,
                self.mipMapBias,
                self.originalDataFormat,
                self.originalDataPos,
                self.originalDataSize,
                self.nativeDataFormat,
                self.nativeDataSize,
                self.nativeDataPos,
                0,
              # self.gx2Texture
            )
        ))

    def fromGX2Texture(self, texture: GX2Texture) -> None:
        self.width = texture.surface.width
        self.height = texture.surface.height
        self.tileMode = texture.surface.tileMode
        self.swizzle = texture.surface.swizzle >> 8 & 7
        self.alignment = texture.surface.alignment
        self.pitch = texture.surface.pitch
        self.mipLevel = texture.surface.numMips
        self.compSel = texture.compSel
        self.mipOffset = tuple((texture.surface.mipOffset[i] if i < 3 else 0) for i in range(13))
        self.nativeDataFormat = TextureFormat[texture.surface.format.name]
        if self.originalDataFormat == TextureFormat.Unorm_RGBA8:
            self.originalDataSize = self.width * self.height * 4
        elif self.originalDataFormat == TextureFormat.Unorm_RGB8:
            self.originalDataSize = self.width * self.height * 3
        else:
            self.originalDataSize = 0

    def fromImage(self, texture: Image.Image) -> None:
        self.width = texture.width
        self.height = texture.height
        self.compSel = 0x00010203 if texture.mode == 'RGBA' else 0x00010205
        self.originalDataFormat = TextureFormat.Unorm_RGBA8 if texture.mode == 'RGBA' else TextureFormat.Unorm_RGB8


class nw__eft__AnimKeyFrameKey:
    structSize = VEC2_SIZE

    x: int
    y: Numeric

    def load(self, data: ByteString, pos: int = 0) -> None:
        x: float
        (
            x,      # Time
            self.y  # Value
        ) = struct.unpack_from(VEC2_FMT, data, pos)
        assert x.is_integer()
        self.x = int(x)

        # TODO: Y value verification based on animation target

    def save(self) -> bytes:
        assert self.x == int(self.x)
        return struct.pack(
            VEC2_FMT,
            self.x,
            self.y
        )

    def toDict(self) -> Dict[Literal['time', 'value'], Any]:
        return {
            "time": self.x,
            "value": self.y
        }

    def fromDict(self, dic: Dict[Literal['time', 'value'], Any]) -> None:
        self.x = VerifyS32Positive(dic["time"])

        # TODO: Y value verification based on animation target
        self.y = VerifyF32(dic["value"])


class nw__eft__AnimKeyFrameInfo:
    keys: List[nw__eft__AnimKeyFrameKey]
    interpolation: AnimKeyFrameInterpolationType
    target: AnimKeyFrameApplyType
    isLoop: bool
    reserved: int

    def load(self, data: ByteString, pos: int = 0) -> None:
        basePos = pos

        (
            keyNum,         # Key count
            interpolation,  # Interpolation method
            target,         # Animation target
            isLoop,         # Should loop?
            offset,         # Next entry (offset from beginning of this structure)
            self.reserved   # I think
        ) = struct.unpack_from('>6I', data, pos); pos += struct.calcsize('>6I')

        self.interpolation = AnimKeyFrameInterpolationType(interpolation)
        self.target = AnimKeyFrameApplyType(target)

        assert isLoop in (0, 1)
        self.isLoop = bool(isLoop)

        self.keys = [nw__eft__AnimKeyFrameKey() for _ in range(keyNum)]
        for key in self.keys:
            key.load(data, pos); pos += nw__eft__AnimKeyFrameKey.structSize

        assert pos - basePos == offset

    def getNextEntryOffset(self) -> int:
        return struct.calcsize('>6I') + len(self.keys) * nw__eft__AnimKeyFrameKey.structSize

    def save(self) -> bytes:
        return b''.join((
            struct.pack(
                '>6I',
                len(self.keys),
                self.interpolation,
                self.target,
                int(self.isLoop),
                self.getNextEntryOffset(),
                self.reserved
            ),
            *(key.save() for key in self.keys)
        ))

    def toDict(self) -> DictGeneric:
        return {
            "interpolationType": {
                "description": "Type of interpolation to use when calculating values between keyframes. Possible types are:\n" \
                               "Linear: Linear interpolation\n" \
                               "Smooth: Smoothstep interpolation",
                "value": self.interpolation.name
            },
            "target": {
                "description": "The target of this animation. Possible targets are:\n" \
                               "EmitterFormScale_x: `emitter.shape.extentScale` attribute (x axis)\n" \
                               "EmitterFormScale_y: `emitter.shape.extentScale` attribute (y axis)\n" \
                               "EmitterFormScale_z: `emitter.shape.extentScale` attribute (z axis)\n" \
                               "EmitterScale_x:     `emitter.transform.scale` attribute (x axis)\n" \
                               "EmitterScale_y:     `emitter.transform.scale` attribute (y axis)\n" \
                               "EmitterScale_z:     `emitter.transform.scale` attribute (z axis)\n" \
                               "EmitterRot_x:       `emitter.transform.rot` attribute (x axis)\n" \
                               "EmitterRot_y:       `emitter.transform.rot` attribute (y axis)\n" \
                               "EmitterRot_z:       `emitter.transform.rot` attribute (z axis)\n" \
                               "EmitterTrans_x:     `emitter.transform.trans` attribute (x axis)\n" \
                               "EmitterTrans_y:     `emitter.transform.trans` attribute (y axis)\n" \
                               "EmitterTrans_z:     `emitter.transform.trans` attribute (z axis)\n" \
                               "GlobalColor0_r:     `emitter.globalColor0` attribute (1st element / red)\n" \
                               "GlobalColor0_g:     `emitter.globalColor0` attribute (2nd element / green)\n" \
                               "GlobalColor0_b:     `emitter.globalColor0` attribute (3rd element / blue)\n" \
                               "GlobalColor1_r:     `emitter.globalColor1` attribute (1st element / red)\n" \
                               "GlobalColor1_g:     `emitter.globalColor1` attribute (2nd element / green)\n" \
                               "GlobalColor1_b:     `emitter.globalColor1` attribute (3rd element / blue)\n" \
                               "GlobalAlpha:        `emitter.globalAlpha` attribute\n" \
                               "EmitterRate:        `emission.timing.interval.paramTime.emitRate` attribute\n" \
                               "FigureVel:          `emission.posAndInitVel.figureVel` attribute\n" \
                               "DirVel:             `emission.posAndInitVel.emitterVel` attribute\n" \
                               "Life:               `particle.lifespan` attribute\n" \
                               "ParticleScale_x:    `ptclScale.baseScale` attribute (x axis)\n" \
                               "ParticleScale_y:    `ptclScale.baseScale` attribute (y axis)",
                "value": self.target.name
            },
            "isLoop": {
                "description": "Should the animation loop?",
                "value": self.isLoop
            },
            "reserved": {
                "description": "Useless integer value",
                "value": self.reserved
            },
            "keys": [key.toDict() for key in self.keys]
        }

    def fromDict(self, dic: DictGeneric) -> None:
        self.interpolation = AnimKeyFrameInterpolationType[dic["interpolationType"]["value"]]
        self.target = AnimKeyFrameApplyType[dic["target"]["value"]]
        self.isLoop = VerifyBool(dic["isLoop"]["value"])
        self.reserved = VerifyU32(dic["reserved"]["value"])
        self.keys = []
        for keyDic in dic["keys"]:
            key = nw__eft__AnimKeyFrameKey()
            key.fromDict(keyDic)
            self.keys.append(key)


class nw__eft__AnimKeyFrameInfoArray:
    anims: List[nw__eft__AnimKeyFrameInfo]

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            magic,      # Magic code ('KEYA': KEYframe animation Array)
            numAnims    # Number of AnimKeyFrameInfo entries
        ) = struct.unpack_from('>4sI', data, pos); pos += struct.calcsize('>4sI')

        assert magic == b'KEYA'

        self.anims = [nw__eft__AnimKeyFrameInfo() for _ in range(numAnims)]
        for info in self.anims:
            info.load(data, pos); pos += info.getNextEntryOffset()

    def save(self) -> bytes:
        return b''.join((
            struct.pack(
                '>4sI',
                b'KEYA',
                len(self.anims)
            ),
            *(info.save() for info in self.anims)
        ))

    def toDict(self) -> Dict[Literal['animations'], List[DictGeneric]]:
        return {"animations": [info.toDict() for info in self.anims]}

    def fromDict(self, dic: Dict[Literal['animations'], List[DictGeneric]]) -> None:
        self.anims = []
        for infoDic in dic["animations"]:
            info = nw__eft__AnimKeyFrameInfo()
            info.fromDict(infoDic)
            self.anims.append(info)

    def toYAML(self, file_path: str) -> None:
        dictArray = self.toDict()
        with open(file_path, 'w', encoding='utf-8') as outf:
            yaml.dump(dictArray, outf, yaml.CSafeDumper, default_flow_style=False, sort_keys=False)

    def fromYAML(self, file_path: str) -> None:
        with open(file_path, encoding='utf-8') as inf:
            dictArray = yaml.load(inf, yaml.CSafeLoader)
        self.fromDict(dictArray)


class nw__eft__AnimKeyTable:
    structSize = struct.calcsize('>4x2I')
    assert structSize == 0xC

    animKeyTable: Optional[nw__eft__AnimKeyFrameInfoArray]
    animPos: int
    dataSize: int

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
          # self.animKeyTable,  # Animation table (set at runtime)
            self.animPos,       # Keyframe animation data (offset from beginning of keyframe animation table)
            self.dataSize       # Animation data size
        ) = struct.unpack_from('>4x2I', data, pos)

        self.animKeyTable = None

    def save(self) -> bytes:
        return struct.pack(
            '>4x2I',
          # self.animKeyTable,
            self.animPos,
            self.dataSize
        )


class nw__eft__PrimitiveFigure:
    structSize = struct.calcsize('>4x2I')
    assert structSize == 0xC

    dataSize: int
    index: int

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
          # self.primitiveTableInfo,    # Primitive table (Not even set at runtime, this thing is straight up useless)
            self.dataSize,              # Primitive table size
            self.index                  # Index
        ) = struct.unpack_from('>4x2I', data, pos)

    def save(self) -> bytes:
        return struct.pack(
            '>4x2I',
          # self.primitiveTableInfo,
            self.dataSize,
            self.index
        )

    def toDict(self) -> DictGeneric:
        return {
            "description": "Primitive model to use for the figure of particles",
            "filename": {
                "description": "Filename of primitive to use, relative to the primitive path set in the project YAML file",
                "value": None if self.index == 0xFFFFFFFF else f'{self.index}.glb'
            }
        }

    def fromDict(self, dic: DictGeneric) -> None:
        self.dataSize = 0
        self.index = ResolvePrimitiveIndex(dic["filename"]["value"])


class nw__eft__TextureEmitterData:
    structSize = (
        struct.calcsize('>6B2x2h') +
        struct.calcsize(f'>{SystemConstants.TexturePatternNum}B') +
        struct.calcsize('>I2fI') +
        VEC2_SIZE * 6 +
        struct.calcsize('>3f')
    )
    assert structSize == 0x78

    texPtnAnimMode: TexturePatternAnimMode
    isTexPatAnimRand: bool
    numTexDivX: int
    numTexDivY: int
    numTexPat: int
    texPatFreq: int
    texPatTblUse: int
    texPatTbl: Tuple[
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int
    ]
    texAddressingMode: TextureAddressing
    uvShiftAnimMode: UvShiftAnimMode
    uvScroll: UvNumeric
    uvScrollInit: UvNumeric
    uvScrollInitRand: UvNumeric
    uvScale: UvNumeric
    uvScaleInit: UvNumeric
    uvScaleInitRand: UvNumeric
    uvRot: Numeric
    uvRotInit: Numeric
    uvRotInitRand: Numeric

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            isTexPatAnim,       # Has texture pattern animation?
            isTexPatAnimRand,   # Randomly start texture pattern animation?
            isTexPatAnimClump,  # Clamp texture pattern animation?
            self.numTexDivX,    # Number of horizontal divisions
            self.numTexDivY,    # Number of vertical divisions
            self.numTexPat,     # Number of texture patterns
            self.texPatFreq,    # Texture pattern animation period
            self.texPatTblUse   # Number of values used from pattern table
        ) = struct.unpack_from('>6B2x2h', data, pos); pos += struct.calcsize('>6B2x2h')

        assert isTexPatAnim in (0, 1)

        assert isTexPatAnimRand in (0, 1)
        self.isTexPatAnimRand = bool(isTexPatAnimRand)

        assert isTexPatAnimClump in (0, 1)

        assert self.numTexDivX > 0
        assert self.numTexDivY > 0
        assert 1 <= self.texPatTblUse < 32

        if self.texPatFreq == 0:
            assert not isTexPatAnimClump
            if self.numTexPat > 1:
                self.texPtnAnimMode = TexturePatternAnimMode.Random
                assert not isTexPatAnim
                assert not self.isTexPatAnimRand
            elif isTexPatAnim:
                self.texPtnAnimMode = TexturePatternAnimMode.LifeFit
                assert self.numTexPat == 1
                assert not self.isTexPatAnimRand
            else:
                self.texPtnAnimMode = TexturePatternAnimMode.Null
                assert self.numTexPat == 1
        else:
            assert self.texPatFreq > 0 and isTexPatAnim and self.numTexPat == 1
            if isTexPatAnimClump:
                self.texPtnAnimMode = TexturePatternAnimMode.Clamp
                assert not self.isTexPatAnimRand
            else:
                self.texPtnAnimMode = TexturePatternAnimMode.Loop

        # Texture pattern table
        texPatTbl_fmt = f'>{SystemConstants.TexturePatternNum}B'
        self.texPatTbl = struct.unpack_from(texPatTbl_fmt, data, pos); pos += struct.calcsize(texPatTbl_fmt)

        (
            texAddressingMode,  # Texture addressing mode
            texUScale,          # U-direction scale
            texVScale,          # V-direction scale
            uvShiftAnimMode     # UV shift animation mode
        ) = struct.unpack_from('>I2fI', data, pos); pos += struct.calcsize('>I2fI')

        self.texAddressingMode = TextureAddressing(texAddressingMode)
        self.uvShiftAnimMode = UvShiftAnimMode(uvShiftAnimMode)

      # if self.texAddressingMode == TextureAddressing.U2V1:
      #     assert texUScale == F32Standardize(2 / self.numTexDivX)
      #     assert texVScale == F32Standardize(1 / self.numTexDivY)
      # elif self.texAddressingMode == TextureAddressing.U1V2:
      #     assert texUScale == F32Standardize(1 / self.numTexDivX)
      #     assert texVScale == F32Standardize(2 / self.numTexDivY)
      # elif self.texAddressingMode == TextureAddressing.U2V2:
      #     assert texUScale == F32Standardize(2 / self.numTexDivX)
      #     assert texVScale == F32Standardize(2 / self.numTexDivY)
      # else:
      #     assert self.texAddressingMode == TextureAddressing.U1V1
      #     assert texUScale == F32Standardize(1 / self.numTexDivX)
      #     assert texVScale == F32Standardize(1 / self.numTexDivY)
        assert texUScale == F32Standardize((1 + (self.texAddressingMode >> 0 & 1)) / self.numTexDivX)
        assert texVScale == F32Standardize((1 + (self.texAddressingMode >> 1 & 1)) / self.numTexDivY)

        # UV scroll addition value
        self.uvScroll = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # UV scroll initial value
        self.uvScrollInit = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # UV scroll initial value random factor
        self.uvScrollInitRand = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE
        assert self.uvScrollInitRand[0] >= 0.0 and \
               self.uvScrollInitRand[1] >= 0.0

        # UV scale addition value
        self.uvScale = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # UV scale initial value
        self.uvScaleInit = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # UV scale initial value random factor
        self.uvScaleInitRand = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE
        assert self.uvScaleInitRand[0] >= 0.0 and \
               self.uvScaleInitRand[1] >= 0.0

        (
            uvRot,          # UV rotation addition value
            uvRotInit,      # UV rotation initial value
            uvRotInitRand   # UV rotation initial value random factor
        ) = struct.unpack_from('>3f', data, pos)

        self.uvRot = F32StandardToRegularMulti(uvRot, MATH_PI_STD, MATH_PI, -MATH_PI_STD, -MATH_PI)
        assert -MATH_PI <= self.uvRot <= MATH_PI

        self.uvRotInit = F32StandardToRegular(uvRotInit, MATH_PI_2_STD, MATH_PI_2)
        assert 0.0 <= self.uvRotInit <= MATH_PI_2

        self.uvRotInitRand = F32StandardToRegular(uvRotInitRand, MATH_PI_2_STD, MATH_PI_2)
        assert 0.0 <= self.uvRotInitRand <= MATH_PI_2

        if self.uvShiftAnimMode not in (UvShiftAnimMode.Trans, UvShiftAnimMode.All):
            assert self.uvScroll == (0.0, 0.0)
            assert self.uvScrollInit == (0.0, 0.0)
            assert self.uvScrollInitRand == (0.0, 0.0)

        if self.uvShiftAnimMode not in (UvShiftAnimMode.Scale, UvShiftAnimMode.All):
            assert self.uvScale == (0.0, 0.0)
            assert self.uvScaleInit == (1.0, 1.0)
            assert self.uvScaleInitRand == (0.0, 0.0)

        if self.uvShiftAnimMode not in (UvShiftAnimMode.Rot, UvShiftAnimMode.All):
            assert self.uvRot == 0.0
            assert self.uvRotInit == 0.0
            assert self.uvRotInitRand == 0.0

    def save(self) -> bytes:
        if self.uvShiftAnimMode in (UvShiftAnimMode.Trans, UvShiftAnimMode.All):
            uvScroll = self.uvScroll
            uvScrollInit = self.uvScrollInit
            uvScrollInitRand = self.uvScrollInitRand
            assert uvScrollInitRand[0] >= 0.0 and \
                   uvScrollInitRand[1] >= 0.0
        else:
            uvScroll = (0.0, 0.0)
            uvScrollInit = (0.0, 0.0)
            uvScrollInitRand = (0.0, 0.0)

        if self.uvShiftAnimMode in (UvShiftAnimMode.Scale, UvShiftAnimMode.All):
            uvScale = self.uvScale
            uvScaleInit = self.uvScaleInit
            uvScaleInitRand = self.uvScaleInitRand
            assert uvScaleInitRand[0] >= 0.0 and \
                   uvScaleInitRand[1] >= 0.0
        else:
            uvScale = (0.0, 0.0)
            uvScaleInit = (1.0, 1.0)
            uvScaleInitRand = (0.0, 0.0)

        if self.uvShiftAnimMode in (UvShiftAnimMode.Rot, UvShiftAnimMode.All):
            uvRot = self.uvRot
            uvRotInit = self.uvRotInit
            uvRotInitRand = self.uvRotInitRand
            assert -MATH_PI <= uvRot <= MATH_PI
            assert 0.0 <= uvRotInit <= MATH_PI_2
            assert 0.0 <= uvRotInitRand <= MATH_PI_2
        else:
            uvRot = 0.0
            uvRotInit = 0.0
            uvRotInitRand = 0.0

      # if self.texAddressingMode == TextureAddressing.U2V1:
      #     texUScale = 2 / self.numTexDivX
      #     texVScale = 1 / self.numTexDivY
      # elif self.texAddressingMode == TextureAddressing.U1V2:
      #     texUScale = 1 / self.numTexDivX
      #     texVScale = 2 / self.numTexDivY
      # elif self.texAddressingMode == TextureAddressing.U2V2:
      #     texUScale = 2 / self.numTexDivX
      #     texVScale = 2 / self.numTexDivY
      # else:
      #     assert self.texAddressingMode == TextureAddressing.U1V1
      #     texUScale = 1 / self.numTexDivX
      #     texVScale = 1 / self.numTexDivY
        texUScale = (1 + (self.texAddressingMode >> 0 & 1)) / self.numTexDivX
        texVScale = (1 + (self.texAddressingMode >> 1 & 1)) / self.numTexDivY

        texPatFreq = 0
        if self.texPtnAnimMode in (TexturePatternAnimMode.Clamp, TexturePatternAnimMode.Loop):
            texPatFreq = self.texPatFreq
            assert texPatFreq > 0

        numTexPat = 1
        if self.texPtnAnimMode == TexturePatternAnimMode.Random:
            numTexPat = self.numTexPat
            assert numTexPat > 1

        isTexPatAnimRand = False
        if self.texPtnAnimMode in (TexturePatternAnimMode.Loop, TexturePatternAnimMode.Null):
            isTexPatAnimRand = self.isTexPatAnimRand

        isTexPatAnimClump = self.texPtnAnimMode == TexturePatternAnimMode.Clamp
        isTexPatAnim = self.texPtnAnimMode not in (TexturePatternAnimMode.Null, TexturePatternAnimMode.Random)

        assert self.numTexDivX > 0
        assert self.numTexDivY > 0
        assert 1 <= self.texPatTblUse < 32

        return b''.join((
            struct.pack(
                '>6B2x2h',
                int(isTexPatAnim),
                int(isTexPatAnimRand),
                int(isTexPatAnimClump),
                self.numTexDivX,
                self.numTexDivY,
                numTexPat,
                texPatFreq,
                self.texPatTblUse
            ),
            struct.pack(f'>{SystemConstants.TexturePatternNum}B', *self.texPatTbl),
            struct.pack(
                '>I2fI',
                self.texAddressingMode,
                texUScale,
                texVScale,
                self.uvShiftAnimMode
            ),
            struct.pack(VEC2_FMT, *uvScroll),
            struct.pack(VEC2_FMT, *uvScrollInit),
            struct.pack(VEC2_FMT, *uvScrollInitRand),
            struct.pack(VEC2_FMT, *uvScale),
            struct.pack(VEC2_FMT, *uvScaleInit),
            struct.pack(VEC2_FMT, *uvScaleInitRand),
            struct.pack(
                '>3f',
                uvRot,
                uvRotInit,
                uvRotInitRand
            )
        ))


class nw__eft__UserShaderParam:
    structSize = struct.calcsize('>32f')
    assert structSize == 0x80

    param: Tuple[
        Numeric, Numeric, Numeric, Numeric,
        Numeric, Numeric, Numeric, Numeric,
        Numeric, Numeric, Numeric, Numeric,
        Numeric, Numeric, Numeric, Numeric,
        Numeric, Numeric, Numeric, Numeric,
        Numeric, Numeric, Numeric, Numeric,
        Numeric, Numeric, Numeric, Numeric,
        Numeric, Numeric, Numeric, Numeric
    ]

    def load(self, data: ByteString, pos: int = 0) -> None:
        self.param = struct.unpack_from('>32f', data, pos)

    def save(self) -> bytes:
        return struct.pack('>32f', *self.param)

    def toDict(self) -> DictGeneric:
        return {
            "description": "Decimal number array (Size = 32).",
            "value": list(self.param)
        }

    def fromDict(self, dic: DictGeneric) -> None:
        self.param = struct.unpack('>32f', struct.pack('>32f', *(dic["value"])))


def UvShiftAnimToDict(textureData: nw__eft__TextureEmitterData) -> DictGeneric:
    return {
        "mode": {
            "description": "Animation type. Possible types are:\n" \
                           "Null:  Disables texture UV animation.\n" \
                           "Trans: Enables texture UV translation animation.\n" \
                           "Scale: Enables texture UV scale animation.\n" \
                           "Rot:   Enables texture UV rotation animation.\n" \
                           "All:   Enables texture UV SRT animation.",
            "value": textureData.uvShiftAnimMode.name
        },
        "trans": {
            "add": {
                "description": "UV translation speed. This value is added to the UV translation in every frame.",
                "value": UvToDict(textureData.uvScroll)
            },
            "init": {
                "description": "Initial UV translation.",
                "value": UvToDict(textureData.uvScrollInit)
            },
            "initRnd": {
                "description": "Initial UV translation random range size.\n" \
                               "A random decimal value picked from the range `(-initRnd, initRnd]` is added to `init`.\n" \
                               "Value must be positive.",
                "value": UvToDict(textureData.uvScrollInitRand)
            }
        },
        "scale": {
            "add": {
                "description": "UV scale speed. This value is added to the UV scale in every frame.",
                "value": UvToDict(textureData.uvScale)
            },
            "init": {
                "description": "Initial UV scale.",
                "value": UvToDict(textureData.uvScaleInit)
            },
            "initRnd": {
                "description": "Initial UV scale random range size.\n" \
                               "A random decimal value picked from the range `(-initRnd, initRnd]` is added to `init`.\n" \
                               "Value must be positive.",
                "value": UvToDict(textureData.uvScaleInitRand)
            }
        },
        "rot": {
            "add": {
                "description": "UV rotation speed, in radians. This value is added to the UV rotation in every frame.\n" \
                              f"Value must be in the range `[{-MATH_PI}, {MATH_PI}]`.",
                "value": textureData.uvRot
            },
            "init": {
                "description": "Initial UV rotation, in radians.\n" \
                              f"Value must be in the range `[0, {MATH_PI_2}]`.",
                "value": textureData.uvRotInit
            },
            "initRnd": {
                "description": "Initial UV rotation random range size, in radians.\n" \
                               "A random decimal value picked from the range `(-initRnd, initRnd]` is added to `init`.\n" \
                              f"Value must be in the range `[0, {MATH_PI_2}]`.",
                "value": textureData.uvRotInitRand
            }
        }
    }


def UvShiftAnimFromDict(textureData: nw__eft__TextureEmitterData, dic: DictGeneric) -> None:
    dic_trans: DictGeneric = dic["trans"]
    dic_scale: DictGeneric = dic["scale"]
    dic_rot: DictGeneric = dic["rot"]

    textureData.uvShiftAnimMode = UvShiftAnimMode[dic["mode"]["value"]]

    textureData.uvScroll = UvFromDict(dic_trans["add"]["value"])
    textureData.uvScrollInit = UvFromDict(dic_trans["init"]["value"])
    textureData.uvScrollInitRand = UvPositiveFromDict(dic_trans["initRnd"]["value"])

    textureData.uvScale = UvFromDict(dic_scale["add"]["value"])
    textureData.uvScaleInit = UvFromDict(dic_scale["init"]["value"])
    textureData.uvScaleInitRand = UvPositiveFromDict(dic_scale["initRnd"]["value"])

    textureData.uvRot = VerifyF32Range(dic_rot["add"]["value"], -MATH_PI, MATH_PI)
    textureData.uvRotInit = VerifyF32Range(dic_rot["init"]["value"], 0.0, MATH_PI_2)
    textureData.uvRotInitRand = VerifyF32Range(dic_rot["initRnd"]["value"], 0.0, MATH_PI_2)


def TextureToDictEx(
    texRes: nw__eft__TextureRes,
    numTexDivX: int, numTexDivY: int,
    texAddressingMode: TextureAddressing,
    uvDict: Optional[DictGeneric],
    texPtnAnimMode: TexturePatternAnimMode,
    numTexPat: int,
    isTexPatAnimRand: bool,
    texPatFreq: int,
    texPatTblUse: int,
    texPatTbl: Tuple[
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int
    ]
) -> DictGeneric:
    assert texRes.gx2TextureIndex < 0 or texRes.imageIndex < 0

    if texRes.gx2TextureIndex >= 0:
        filename = f'{texRes.gx2TextureIndex}.gtx'
    elif texRes.imageIndex >= 0:
        filename = f'{texRes.imageIndex}.png'
    else:
        filename = None

    if filename is None:
        originalDataFormat = None
    else:
        originalDataFormat = 'RGBA' if texRes.originalDataFormat == TextureFormat.Unorm_RGBA8 else 'RGB'


    return {
        "filename": {
            "description": "Filename of texture to use, relative to the texture path set in the project YAML file.\n" \
                           "(Only GTX and PNG allowed, though GTX specifically is strongly recommended.)",
            "value": filename
        },
        "originalDataFormat": {
            "description": "Hint of the format of the original texture that was used to generate either the GTX or PNG file specified in `filename`. Optional.",
            "value": originalDataFormat
        },
        "numTexDivX": {
            "description": "Number of horizontal divisions. This value divides the texture horizontally into patterns which are used in the texture pattern animation.",
            "value": numTexDivX
        },
        "numTexDivY": {
            "description": "Number of vertical divisions. This value divides the texture vertically into patterns which are used in the texture pattern animation.",
            "value": numTexDivY
        },
        "wrapModeU": {
            "description": "Wrap mode in the U (horizontal) direction. Possible modes are:\n" \
                           "Mirror\n" \
                           "Repeat\n" \
                           "Clamp",
                         # "MirrorOnce",
            "value": texRes.wrapModeU.name
        },
        "wrapModeV": {
            "description": "Wrap mode in the V (vertical) direction. Same modes as `wrapModeU`.",
            "value": texRes.wrapModeV.name
        },
        "repetitionCount": {
            "description": "Texture repetition count in each axis to use when the texture is divided into the patterns for the texture pattern animation.\n" \
                           "Possible values are:\n" \
                           "U1V1: No repetition in the texture. (1x1)\n" \
                           "U2V1: The texture is repeated once in the U (horizontal) direction. (2x1)\n" \
                           "U1V2: The texture is repeated once in the V (vertical) direction. (1x2)\n" \
                           "U2V2: The texture is repeated once in both the U (horizontal) & V (vertical) directions. (1x2).",
            "value": texAddressingMode.name
        },
        "filterMode": {
            "description": "Texture filter mode. Possible modes are:\n" \
                           "Linear: Interpolates between texel colors to make textures appear smooth.\n" \
                           "Near:   Displays each texel distinctly.",
            "value": texRes.filterMode.name
        },
        "enableMipLevel": {
            "description": "Mip LOD level maximum value.\n" \
                           "For textures with mipmaps, this value can be used to specify the range of mipmap levels that are applied.\n" \
                          f"(Valid values are `0.0` to `{ENABLE_MIP_LEVEL_MAX}`.)",
            "value": texRes.enableMipLevel
        },
        "mipMapBias": {
            "description": "Mip LOD level bias value.\n" \
                           "For textures with mipmaps, this value can be used to adjust the mipmap levels that are applied.\n" \
                          f"(Valid values are `-32.0` to `{MIP_MAP_BIAS_MAX}`.)",
            "value": texRes.mipMapBias
        },
        **({"texSRTAnim": uvDict} if uvDict is not None else {}),
        "texPatAnim": {
            "mode": {
                "description": "Animation type. Possible types are:\n" \
                               "Null:    Disables texture pattern animation.\n" \
                               "LifeFit: Fits the texture pattern animation to the particle lifespan such that it plays only once and in full.\n" \
                               "Clamp:   Plays the texture pattern animation once, then the final pattern is shown for the rest of the particle lifespan.\n" \
                               "Loop:    Plays the texture pattern animation once, then keeps looping it repeatedly for the rest of the particle lifespan.\n" \
                               "Random:  Plays a single pattern for the entirety of the particle lifespan, randomly selected (per particle).",
                "value": texPtnAnimMode.name
            },
            "numTexPat": {
                "description": "For the `Random` mode, the number of sequential patterns to consider as candidates for random selection.\n" \
                               "Value must be greater than `1` when the `Random` mode is selected.",
                "value": numTexPat
            },
            "isTexPatAnimRand": {
                "description": "For the `Loop` mode, set animation start position randomly (per particle)?\n" \
                               "In other words, if enabled, instead of the animation starting from the first pattern, the animation will start from a random point with it.",
                "value": isTexPatAnimRand
            },
            "texPatFreq": {
                "description": "For the `Clamp` and `Loop` modes, the period of each pattern, in frames.\n" \
                               "Value must be greater than `0` when the `Clamp` and `Loop` modes are selected.",
                "value": texPatFreq
            },
            "texPatTblUse": {
                "description": "For the `LifeFit`, `Clamp` and `Loop` modes, the total number of patterns to use in the animation.\n" \
                               "Value must be in the range `[1, 32)`.",
                "value": texPatTblUse
            },
            "texPatTbl": {
                "description": "For the `LifeFit`, `Clamp` and `Loop` modes, the order of patterns in the animation. (Max size = 32)",
                "value": list(texPatTbl)
            }
        }
    }


def TextureFromDictEx(texRes: nw__eft__TextureRes, dic: DictGeneric) -> Tuple[
    int, int,
    TextureAddressing,
    Optional[DictGeneric],
    TexturePatternAnimMode,
    int,
    bool,
    int,
    int,
    Tuple[
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int
    ]
]:
    texRes.imageIndex, texRes.gx2TextureIndex, texRes.originalDataFormat = ResolveTextureIndex(dic["filename"]["value"], dic["originalDataFormat"]["value"])

    texRes.wrapModeU = TextureWrapMode[dic["wrapModeU"]["value"]]
    texRes.wrapModeV = TextureWrapMode[dic["wrapModeV"]["value"]]
    texRes.filterMode = TextureFilterMode[dic["filterMode"]["value"]]
    texRes.enableMipLevel = VerifyF32Range(dic["enableMipLevel"]["value"], 0.0, ENABLE_MIP_LEVEL_MAX)
    texRes.mipMapBias = VerifyF32Range(dic["mipMapBias"]["value"], -32.0, MIP_MAP_BIAS_MAX)

    try:
        uvDict = dic["texSRTAnim"]
    except KeyError:
        uvDict = None
    else:
        assert uvDict is not None

    dic_texPatAnim = dic["texPatAnim"]

    texPtnAnimMode = TexturePatternAnimMode[dic_texPatAnim["mode"]["value"]]

    texPatFreq = 0
    if texPtnAnimMode in (TexturePatternAnimMode.Clamp, TexturePatternAnimMode.Loop):
        texPatFreq = VerifyS16PositiveNonZero(dic_texPatAnim["texPatFreq"]["value"])

    numTexPat = 1
    if texPtnAnimMode == TexturePatternAnimMode.Random:
        numTexPat = VerifyIntRange(dic_texPatAnim["numTexPat"]["value"], 2, 0xFF)

    isTexPatAnimRand = False
    if texPtnAnimMode in (TexturePatternAnimMode.Loop, TexturePatternAnimMode.Null):
        isTexPatAnimRand = VerifyBool(dic_texPatAnim["isTexPatAnimRand"]["value"])

    texPatTblBase = dic_texPatAnim["texPatTbl"]["value"]
    if texPatTblBase is None:
        texPatTbl = (0,) * SystemConstants.TexturePatternNum
    else:
        assert isinstance(texPatTblBase, list) and len(texPatTblBase) <= SystemConstants.TexturePatternNum
        texPatTbl = tuple((VerifyU8(texPatTblBase[i]) if i < len(texPatTblBase) else 0) for i in range(SystemConstants.TexturePatternNum))

    return (
        VerifyU8PositiveNonZero(dic["numTexDivX"]["value"]), VerifyU8PositiveNonZero(dic["numTexDivY"]["value"]),
        TextureAddressing[dic["repetitionCount"]["value"]],
        uvDict,
        texPtnAnimMode,
        numTexPat,
        isTexPatAnimRand,
        texPatFreq,
        VerifyIntRange(dic_texPatAnim["texPatTblUse"]["value"], 1, 31),
        texPatTbl
    )

class nw__eft__ChildData:
    childColor0Inherit: bool
    childAlphaInherit: bool
    childScaleInherit: bool
    childRotInherit: bool
    childVelInherit: bool
    childEmitterFollow: bool
    childWorldField: bool
    childParentField: bool
    childDrawBeforeParent: bool
    isChildTexPatAnimRand: bool
    childColor1Inherit: bool
    childTextureColorSource: ColorSource
    childPrimitiveColorSource: ColorSource
    childTextureAlphaSource: AlphaSource
    childPrimitiveAlphaSource: AlphaSource
    childEmitRate: int
    childEmitTiming: int
    childLife: int
    childEmitStep: int
    childVelInheritRate: Numeric
    childFigureVel: Numeric
    childRandVel: Vec3Numeric
    childInitPosRand: Numeric
    childPrimitiveFigure: nw__eft__PrimitiveFigure
    childDynamicsRandom: Numeric
    childBlendType: BlendType
    childMeshType: MeshType
    childBillboardType: BillboardType
    childZBufATestType: ZBufATestType
    childTex: nw__eft__TextureRes
    childDisplaySide: DisplaySideType
    childColor0: Color3Numeric
    childColor1: Color3Numeric
    childColorScale: Numeric
    primitiveColorBlend: ColorAlphaBlendType
    primitiveAlphaBlend: ColorAlphaBlendType
    childAlpha: Numeric
    childAlphaTarget: Numeric
    childAlphaInit: Numeric
    childScaleInheritRate: Numeric
    childScale: Vec2Numeric
    childScaleRand: Numeric
    childRotType: PtclRotType
    childInitRot: Vec3Numeric
    childInitRotRand: Vec3Numeric
    childRotVel: Vec3Numeric
    childRotVelRand: Vec3Numeric
    childRotRegist: Numeric
    childRotBasis: Vec2Numeric
    childGravity: Vec3Numeric
    childAlphaStartFrame: int
    childAlphaBaseFrame: int
    childScaleStartFrame: int
    childScaleTarget: Vec2Numeric
    childNumTexPat: int
    childNumTexDivX: int
    childNumTexDivY: int
    childTexAddressingMode = TextureAddressing
    childTexPtnAnimMode: TexturePatternAnimMode
    childTexPatTbl: Tuple[
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int,
        int, int, int, int
    ]
    childTexPatFreq: int
    childTexPatTblUse: int
    childCombinerType: ColorCombinerType
    childAlphaBaseCombinerType: AlphaBaseCombinerType
    childAlphaCommonSource: AlphaCommonSource
    childAirRegist: Numeric
    childShaderType: FragmentShaderVariation
    childUserShaderSetting: int
    childShaderUseSoftEdge: bool
    childShaderParam0: Numeric
    childShaderParam1: Numeric
    childSoftFadeDistance: Numeric
    childSoftVolumeParam: Numeric
    childUserShaderDefine1: str
    childUserShaderDefine2: str
    childUserShaderFlag: int
    childUserShaderSwitchFlag: int
    childUserShaderParam: nw__eft__UserShaderParam

    def __init__(self) -> None:
        self.childPrimitiveFigure = nw__eft__PrimitiveFigure()
        self.childTex = nw__eft__TextureRes()
        self.childUserShaderParam = nw__eft__UserShaderParam()

    def load(self, data: ByteString, childFlg: int, pos: int = 0) -> int:
        basePos = pos

        assert (childFlg & ChildFlg.Enable) != 0
        self.childColor0Inherit = (childFlg & ChildFlg.InheritColor0) != 0
        self.childAlphaInherit = (childFlg & ChildFlg.InheritAlpha) != 0
        self.childScaleInherit = (childFlg & ChildFlg.InheritScale) != 0
        self.childRotInherit = (childFlg & ChildFlg.InheritRot) != 0
        self.childVelInherit = (childFlg & ChildFlg.InheritVel) != 0
        self.childEmitterFollow = (childFlg & ChildFlg.EmitterFollow) != 0
        assert (childFlg & ChildFlg.DispParent) == 0  # Never used
        self.childWorldField = (childFlg & ChildFlg.WorldField) != 0  # Never used, but not constant either
        assert (childFlg & ChildFlg.IsPolygon) == 0  # Never used
        assert (childFlg & ChildFlg.IsEmitterBillboardMtx) == 0  # Never used
        self.childParentField = (childFlg & ChildFlg.ParentField) != 0
        self.childDrawBeforeParent = (childFlg & ChildFlg.PreChildDraw) != 0
        isChildTexPatAnim = (childFlg & ChildFlg.IsTexPatAnim) != 0
        self.isChildTexPatAnimRand = (childFlg & ChildFlg.IsTexPatAnimRand) != 0
        self.childColor1Inherit = (childFlg & ChildFlg.InheritColor1) != 0
        assert (childFlg & ChildFlg.InheritColorScale) == 0  # Never used
        self.childTextureColorSource = ColorSource(int((childFlg & ChildFlg.TextureColorOne) != 0))      # Replace Texture color with 1.0?
        self.childPrimitiveColorSource = ColorSource(int((childFlg & ChildFlg.PrimitiveColorOne) != 0))  # Replace Primitive color with 1.0?
        self.childTextureAlphaSource = AlphaSource(int((childFlg & ChildFlg.TextureAlphaOne) != 0))      # Replace Texture alpha with 1.0?
        self.childPrimitiveAlphaSource = AlphaSource(int((childFlg & ChildFlg.PrimitiveAlphaOne) != 0))  # Replace Primitive alpha with 1.0?
        assert (childFlg & 0b11111111111000000000000000000000) == 0  # Ununsed bits

        (
            self.childEmitRate,         # Emission rate
            self.childEmitTiming,       # Emission start frame (%)
            self.childLife,             # Particle life
            self.childEmitStep,         # Emission interval
            self.childVelInheritRate,   # Velocity inheritance rate
            self.childFigureVel         # Omnidirectional velocity
        ) = struct.unpack_from('>4i2f', data, pos); pos += struct.calcsize('>4i2f')

        assert self.childEmitRate >= 0
        assert 0 <= self.childEmitTiming <= 100
        assert self.childLife > 0
        assert self.childEmitStep >= 0

        # Initial velocity randomizer factor
        self.childRandVel = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        assert self.childRandVel[0] >= 0.0 and \
               self.childRandVel[1] >= 0.0 and \
               self.childRandVel[2] >= 0.0

        # Initial position randomizer factor
        self.childInitPosRand = struct.unpack_from(F32_FMT, data, pos)[0]; pos += F32_SIZE
        assert self.childInitPosRand >= 0.0

        # Primitive to use
        self.childPrimitiveFigure.load(data, pos); pos += nw__eft__PrimitiveFigure.structSize

        (
            self.childDynamicsRandom,   # Dynamics random factor
            childBlendType,             # Blend type
            childMeshType,              # Mesh type
            childBillboardType,         # Billboard type
            childZBufATestType          # Z-buffer, alpha test type
        ) = struct.unpack_from('>f4I', data, pos); pos += struct.calcsize('>f4I')

        assert self.childDynamicsRandom >= 0.0

        self.childBlendType = BlendType(childBlendType)

        self.childMeshType = MeshType(childMeshType)
        if self.childMeshType == MeshType.Primitive:
            assert self.childPrimitiveFigure.index != 0xFFFFFFFF
        else:
            assert self.childPrimitiveFigure.index == 0xFFFFFFFF
            assert self.childMeshType == MeshType.Particle

        assert childBillboardType not in (BillboardType.Primitive, BillboardType.HistoricalStripe, BillboardType.ConsolidatedStripe)
        self.childBillboardType = BillboardType(childBillboardType)

        self.childZBufATestType = ZBufATestType(childZBufATestType)

        # Child texture
        self.childTex.load(data, pos); pos += nw__eft__TextureRes.structSize

        # Display side
        self.childDisplaySide = DisplaySideType(struct.unpack_from(U32_FMT, data, pos)[0]); pos += U32_SIZE

        # Color 0
        self.childColor0 = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Color 1
        self.childColor1 = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        (
            self.childColorScale,       # Color scale
            primitiveColorBlend,        # Primitive color composite type
            primitiveAlphaBlend,        # Primitive alpha composite type
            self.childAlpha,            # Particle alpha 3-value 4-key animation: alpha2
            self.childAlphaTarget,      # Particle alpha 3-value 4-key animation: alpha3
            self.childAlphaInit,        # Particle alpha 3-value 4-key animation: Alpha initial value (alpha1)
            self.childScaleInheritRate  # Scale inheritance rate
        ) = struct.unpack_from('>f2I4f', data, pos); pos += struct.calcsize('>f2I4f')

        assert self.childColorScale >= 0.0
        assert 0.0 <= self.childAlpha <= 1.0
        assert 0.0 <= self.childAlphaTarget <= 1.0
        assert 0.0 <= self.childAlphaInit <= 1.0
        assert self.childScaleInheritRate >= 0.0

        self.primitiveColorBlend = ColorAlphaBlendType(primitiveColorBlend)
        self.primitiveAlphaBlend = ColorAlphaBlendType(primitiveAlphaBlend)

        # Particle scale: Scale base value
        self.childScale = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE
        assert self.childScale[0] >= 0.0 and \
               self.childScale[1] >= 0.0

        (
            self.childScaleRand,    # Particle scale: Scale randomizer factor
            childRotType            # Rotation type
        ) = struct.unpack_from('>fI', data, pos); pos += struct.calcsize('>fI')

        assert 0.0 <= self.childScaleRand <= 1.0

        self.childRotType = PtclRotType(childRotType)

        # Particle rotation: Initial rotation
        childInitRotX: float
        childInitRotY: float
        childInitRotZ: float
        (
            childInitRotX,
            childInitRotY,
            childInitRotZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Particle rotation: Initial rotation random factor
        childInitRotRandX: float
        childInitRotRandY: float
        childInitRotRandZ: float
        (
            childInitRotRandX,
            childInitRotRandY,
            childInitRotRandZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Particle rotation: Rotation velocity
        childRotVelX: float
        childRotVelY: float
        childRotVelZ: float
        (
            childRotVelX,
            childRotVelY,
            childRotVelZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Particle rotation: Rotation velocity random factor
        childRotVelRandX: float
        childRotVelRandY: float
        childRotVelRandZ: float
        (
            childRotVelRandX,
            childRotVelRandY,
            childRotVelRandZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Particle rotation: Rotation velocity decay factor
        self.childRotRegist = struct.unpack_from(F32_FMT, data, pos)[0]; pos += F32_SIZE
        assert self.childRotRegist >= 0.0

        # Particle rotation: Rotation velocity decay factor
        self.childRotBasis = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # Gravity
        self.childGravity = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        (
            self.childAlphaStartFrame,  # Particle alpha 3-value 4-key animation: Second section length (in frames)
            self.childAlphaBaseFrame,   # Particle alpha 3-value 4-key animation: First section length (in frames)
            self.childScaleStartFrame   # Particle scale animation: Start frame
        ) = struct.unpack_from('>3i', data, pos); pos += struct.calcsize('>3i')

        assert 0 <= self.childAlphaStartFrame <= self.childLife
        assert 0 <= self.childAlphaBaseFrame <= self.childLife
        assert 0 <= self.childScaleStartFrame <= self.childLife

        # Particle scale animation: Addition target value (Addition base value is 1)
        self.childScaleTarget = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE
        assert self.childScaleTarget[0] >= 0.0 and \
               self.childScaleTarget[1] >= 0.0

        (
            self.childNumTexPat,    # Number of texture patterns
            self.childNumTexDivX,   # Number of horizontal divisions
            self.childNumTexDivY,   # Number of vertical divisions
            childTexUScale,         # U-direction scale
            childTexVScale          # V-direction scale
        ) = struct.unpack_from('>H2B2f', data, pos); pos += struct.calcsize('>H2B2f')

        assert self.childNumTexDivX > 0
        assert self.childNumTexDivY > 0
        assert self.childNumTexPat <= 0xFF

        childTexAddressingModeU = round(childTexUScale * self.childNumTexDivX)
        childTexAddressingModeV = round(childTexVScale * self.childNumTexDivY)
      # if childTexAddressingModeU == 2 and childTexAddressingModeV == 1:
      #     self.childTexAddressingMode = TextureAddressing.U2V1
      # elif childTexAddressingModeU == 1 and childTexAddressingModeV == 2:
      #     self.childTexAddressingMode = TextureAddressing.U1V2
      # elif childTexAddressingModeU == 2 and childTexAddressingModeV == 2:
      #     self.childTexAddressingMode = TextureAddressing.U2V2
      # else:
      #     assert childTexAddressingModeU == 1 and childTexAddressingModeV == 1
      #     self.childTexAddressingMode = TextureAddressing.U1V1
        assert childTexAddressingModeU in (1, 2)
        assert childTexAddressingModeV in (1, 2)
        self.childTexAddressingMode = TextureAddressing(
            (childTexAddressingModeU - 1) << 0 |
            (childTexAddressingModeV - 1) << 1
        )

        # Texture pattern table
        childTexPatTbl_fmt = f'>{SystemConstants.TexturePatternNum}B'
        self.childTexPatTbl = struct.unpack_from(childTexPatTbl_fmt, data, pos); pos += struct.calcsize(childTexPatTbl_fmt)

        (
            self.childTexPatFreq,                   # Texture pattern animation period
            self.childTexPatTblUse,                 # Number of values used from pattern table
            isChildTexPatAnimClump,                 # Clamp texture pattern animation?
            childCombinerType,                      # Combiner type
            childAlphaCombinerType,                 # Alpha combiner type
            self.childAirRegist,                    # Air resistance
            childShaderType,                        # Shader type
            self.childUserShaderSetting,            # User shader type
            childShaderUseSoftEdge,                 # Use soft edge?
            childShaderApplyAlphaToRefract,         # Apply alpha to refraction shader? (Useless since there isn't a second texture)
            self.childShaderParam0,                 # Shader parameter 0
            self.childShaderParam1,                 # Shader parameter 1
            self.childSoftFadeDistance,             # Distance to soft edge/fade
            self.childSoftVolumeParam,              # Soft edge/volume value
            childUserShaderDefine1,                 # User shader compiler define 1
            childUserShaderDefine2,                 # User shader compiler define 2
            self.childUserShaderFlag,               # User shader flags
            self.childUserShaderSwitchFlag          # User shader switch flags
        ) = struct.unpack_from('>2hB3x2If4B4f16s16s2I', data, pos); pos += struct.calcsize('>2hB3x2If4B4f16s16s2I')

        assert isChildTexPatAnimClump in (0, 1)

        assert 1 <= self.childTexPatTblUse < 32

        if self.childTexPatFreq == 0:
            assert not isChildTexPatAnimClump
            if self.childNumTexPat > 1:
                self.childTexPtnAnimMode = TexturePatternAnimMode.Random
                assert not isChildTexPatAnim
                assert not self.isChildTexPatAnimRand
            elif isChildTexPatAnim:
                self.childTexPtnAnimMode = TexturePatternAnimMode.LifeFit
                assert self.childNumTexPat == 1
                assert not self.isChildTexPatAnimRand
            else:
                self.childTexPtnAnimMode = TexturePatternAnimMode.Null
                assert self.childNumTexPat == 1
        else:
            assert self.childTexPatFreq > 0 and isChildTexPatAnim and self.childNumTexPat == 1
            if isChildTexPatAnimClump:
                self.childTexPtnAnimMode = TexturePatternAnimMode.Clamp
                assert not self.isChildTexPatAnimRand
            else:
                self.childTexPtnAnimMode = TexturePatternAnimMode.Loop

        self.childCombinerType = ColorCombinerType(childCombinerType)
        self.childAlphaBaseCombinerType, self.childAlphaCommonSource = AlphaCombinerType(childAlphaCombinerType).deconstruct()

        assert self.childAirRegist >= 0.0

        assert childShaderType != FragmentShaderVariation.Distortion
        self.childShaderType = FragmentShaderVariation(childShaderType)

        assert 0 <= self.childUserShaderSetting < 8

        assert childShaderUseSoftEdge in (0, 1)
        self.childShaderUseSoftEdge = bool(childShaderUseSoftEdge)

        assert childShaderApplyAlphaToRefract == 1

        assert self.childShaderParam0 >= 0.0
        assert self.childShaderParam1 >= 0.0
        assert self.childSoftFadeDistance >= 0.0
        assert self.childSoftVolumeParam >= 0.0

        if self.childRotType == PtclRotType.NoWork:
            assert childInitRotX     == 0.0 and childInitRotY     == 0.0 and childInitRotZ     == 0.0
            assert childInitRotRandX == 0.0 and childInitRotRandY == 0.0 and childInitRotRandZ == 0.0
            assert childRotVelX      == 0.0 and childRotVelY      == 0.0 and childRotVelZ      == 0.0
            assert childRotVelRandX  == 0.0 and childRotVelRandY  == 0.0 and childRotVelRandZ  == 0.0

        elif self.childRotType == PtclRotType.RotX:
            assert childInitRotY     == 0.0 and childInitRotZ     == 0.0
            assert childInitRotRandY == 0.0 and childInitRotRandZ == 0.0
            assert childRotVelY      == 0.0 and childRotVelZ      == 0.0
            assert childRotVelRandY  == 0.0 and childRotVelRandZ  == 0.0

        elif self.childRotType == PtclRotType.RotY:
            assert childInitRotX     == 0.0 and childInitRotZ     == 0.0
            assert childInitRotRandX == 0.0 and childInitRotRandZ == 0.0
            assert childRotVelX      == 0.0 and childRotVelZ      == 0.0
            assert childRotVelRandX  == 0.0 and childRotVelRandZ  == 0.0

        elif self.childRotType == PtclRotType.RotZ:
            assert childInitRotX     == 0.0 and childInitRotY     == 0.0
            assert childInitRotRandX == 0.0 and childInitRotRandY == 0.0
            assert childRotVelX      == 0.0 and childRotVelY      == 0.0
            assert childRotVelRandX  == 0.0 and childRotVelRandY  == 0.0

        if self.childRotType in (PtclRotType.RotX, PtclRotType.RotXYZ):
            childInitRotX = F32StandardToRegular(childInitRotX, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childInitRotX <= MATH_PI_2

            childInitRotRandX = F32StandardToRegular(childInitRotRandX, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childInitRotRandX <= MATH_PI_2

            childRotVelX = F32StandardToRegularMulti(childRotVelX, MATH_PI_STD, MATH_PI, -MATH_PI_STD, -MATH_PI)
            assert -MATH_PI <= childRotVelX <= MATH_PI

            childRotVelRandX = F32StandardToRegular(childRotVelRandX, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childRotVelRandX <= MATH_PI_2

        if self.childRotType in (PtclRotType.RotY, PtclRotType.RotXYZ):
            childInitRotY = F32StandardToRegular(childInitRotY, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childInitRotY <= MATH_PI_2

            childInitRotRandY = F32StandardToRegular(childInitRotRandY, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childInitRotRandY <= MATH_PI_2

            childRotVelY = F32StandardToRegularMulti(childRotVelY, MATH_PI_STD, MATH_PI, -MATH_PI_STD, -MATH_PI)
            assert -MATH_PI <= childRotVelY <= MATH_PI

            childRotVelRandY = F32StandardToRegular(childRotVelRandY, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childRotVelRandY <= MATH_PI_2

        if self.childRotType in (PtclRotType.RotZ, PtclRotType.RotXYZ):
            childInitRotZ = F32StandardToRegular(childInitRotZ, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childInitRotZ <= MATH_PI_2

            childInitRotRandZ = F32StandardToRegular(childInitRotRandZ, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childInitRotRandZ <= MATH_PI_2

            childRotVelZ = F32StandardToRegularMulti(childRotVelZ, MATH_PI_STD, MATH_PI, -MATH_PI_STD, -MATH_PI)
            assert -MATH_PI <= childRotVelZ <= MATH_PI

            childRotVelRandZ = F32StandardToRegular(childRotVelRandZ, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= childRotVelRandZ <= MATH_PI_2

        self.childInitRot     = (childInitRotX,     childInitRotY,     childInitRotZ)
        self.childInitRotRand = (childInitRotRandX, childInitRotRandY, childInitRotRandZ)
        self.childRotVel      = (childRotVelX,      childRotVelY,      childRotVelZ)
        self.childRotVelRand  = (childRotVelRandX,  childRotVelRandY,  childRotVelRandZ)

        self.childUserShaderDefine1 = readString(childUserShaderDefine1)
        self.childUserShaderDefine2 = readString(childUserShaderDefine2)

        # User shader parameters
        self.childUserShaderParam.load(data, pos); pos += nw__eft__UserShaderParam.structSize

        assert pos - basePos == 0x2FC
        return pos

    def getChildFlg(self) -> int:
        isChildTexPatAnimRand = False
        if self.childTexPtnAnimMode in (TexturePatternAnimMode.Loop, TexturePatternAnimMode.Null):
            isChildTexPatAnimRand = self.isChildTexPatAnimRand

        isChildTexPatAnim = self.childTexPtnAnimMode not in (TexturePatternAnimMode.Null, TexturePatternAnimMode.Random)

        return (
            ChildFlg.Enable |
            self.childColor0Inherit * ChildFlg.InheritColor0 |
            self.childAlphaInherit * ChildFlg.InheritAlpha |
            self.childScaleInherit * ChildFlg.InheritScale |
            self.childRotInherit * ChildFlg.InheritRot |
            self.childVelInherit * ChildFlg.InheritVel |
            self.childEmitterFollow * ChildFlg.EmitterFollow |
            self.childWorldField * ChildFlg.WorldField |
            self.childParentField * ChildFlg.ParentField |
            self.childDrawBeforeParent * ChildFlg.PreChildDraw |
            isChildTexPatAnim * ChildFlg.IsTexPatAnim |
            isChildTexPatAnimRand * ChildFlg.IsTexPatAnimRand |
            self.childColor1Inherit * ChildFlg.InheritColor1 |
            bool(self.childTextureColorSource) * ChildFlg.TextureColorOne |
            bool(self.childPrimitiveColorSource) * ChildFlg.PrimitiveColorOne |
            bool(self.childTextureAlphaSource) * ChildFlg.TextureAlphaOne |
            bool(self.childPrimitiveAlphaSource) * ChildFlg.PrimitiveAlphaOne
        )

    def save(self) -> bytes:
        childUserShaderDefine1 = self.childUserShaderDefine1.encode('shift_jis').ljust(16, b'\0')
        assert len(childUserShaderDefine1) == 16 and childUserShaderDefine1[-1] == 0

        childUserShaderDefine2 = self.childUserShaderDefine2.encode('shift_jis').ljust(16, b'\0')
        assert len(childUserShaderDefine2) == 16 and childUserShaderDefine2[-1] == 0

        assert self.childMeshType != MeshType.Stripe

        if self.childRotType == PtclRotType.NoWork:
            childInitRotX     = 0.0
            childInitRotY     = 0.0
            childInitRotZ     = 0.0
            childInitRotRandX = 0.0
            childInitRotRandY = 0.0
            childInitRotRandZ = 0.0
            childRotVelX      = 0.0
            childRotVelY      = 0.0
            childRotVelZ      = 0.0
            childRotVelRandX  = 0.0
            childRotVelRandY  = 0.0
            childRotVelRandZ  = 0.0

        elif self.childRotType == PtclRotType.RotX:
            childInitRotX     = self.childInitRot[0]
            childInitRotY     = 0.0
            childInitRotZ     = 0.0
            childInitRotRandX = self.childInitRotRand[0]
            childInitRotRandY = 0.0
            childInitRotRandZ = 0.0
            childRotVelX      = self.childRotVel[0]
            childRotVelY      = 0.0
            childRotVelZ      = 0.0
            childRotVelRandX  = self.childRotVelRand[0]
            childRotVelRandY  = 0.0
            childRotVelRandZ  = 0.0

        elif self.childRotType == PtclRotType.RotY:
            childInitRotX     = 0.0
            childInitRotY     = self.childInitRot[1]
            childInitRotZ     = 0.0
            childInitRotRandX = 0.0
            childInitRotRandY = self.childInitRotRand[1]
            childInitRotRandZ = 0.0
            childRotVelX      = 0.0
            childRotVelY      = self.childRotVel[1]
            childRotVelZ      = 0.0
            childRotVelRandX  = 0.0
            childRotVelRandY  = self.childRotVelRand[1]
            childRotVelRandZ  = 0.0

        elif self.childRotType == PtclRotType.RotZ:
            childInitRotX     = 0.0
            childInitRotY     = 0.0
            childInitRotZ     = self.childInitRot[2]
            childInitRotRandX = 0.0
            childInitRotRandY = 0.0
            childInitRotRandZ = self.childInitRotRand[2]
            childRotVelX      = 0.0
            childRotVelY      = 0.0
            childRotVelZ      = self.childRotVel[2]
            childRotVelRandX  = 0.0
            childRotVelRandY  = 0.0
            childRotVelRandZ  = self.childRotVelRand[2]

        else:
            (childInitRotX,     childInitRotY,     childInitRotZ)     = self.childInitRot
            (childInitRotRandX, childInitRotRandY, childInitRotRandZ) = self.childInitRotRand
            (childRotVelX,      childRotVelY,      childRotVelZ)      = self.childRotVel
            (childRotVelRandX,  childRotVelRandY,  childRotVelRandZ)  = self.childRotVelRand

        if self.childRotType in (PtclRotType.RotX, PtclRotType.RotXYZ):
            assert 0.0 <= childInitRotX <= MATH_PI_2
            assert 0.0 <= childInitRotRandX <= MATH_PI_2
            assert -MATH_PI <= childRotVelX <= MATH_PI
            assert 0.0 <= childRotVelRandX <= MATH_PI_2

        if self.childRotType in (PtclRotType.RotY, PtclRotType.RotXYZ):
            assert 0.0 <= childInitRotY <= MATH_PI_2
            assert 0.0 <= childInitRotRandY <= MATH_PI_2
            assert -MATH_PI <= childRotVelY <= MATH_PI
            assert 0.0 <= childRotVelRandY <= MATH_PI_2

        if self.childRotType in (PtclRotType.RotZ, PtclRotType.RotXYZ):
            assert 0.0 <= childInitRotZ <= MATH_PI_2
            assert 0.0 <= childInitRotRandZ <= MATH_PI_2
            assert -MATH_PI <= childRotVelZ <= MATH_PI
            assert 0.0 <= childRotVelRandZ <= MATH_PI_2

      # if self.childTexAddressingMode == TextureAddressing.U2V1:
      #     childTexUScale = 2 / self.childNumTexDivX
      #     childTexVScale = 1 / self.childNumTexDivY
      # elif self.childTexAddressingMode == TextureAddressing.U1V2:
      #     childTexUScale = 1 / self.childNumTexDivX
      #     childTexVScale = 2 / self.childNumTexDivY
      # elif self.childTexAddressingMode == TextureAddressing.U2V2:
      #     childTexUScale = 2 / self.childNumTexDivX
      #     childTexVScale = 2 / self.childNumTexDivY
      # else:
      #     assert self.childTexAddressingMode == TextureAddressing.U1V1
      #     childTexUScale = 1 / self.childNumTexDivX
      #     childTexVScale = 1 / self.childNumTexDivY
        childTexUScale = (1 + (self.childTexAddressingMode >> 0 & 1)) / self.childNumTexDivX
        childTexVScale = (1 + (self.childTexAddressingMode >> 1 & 1)) / self.childNumTexDivY

        childTexPatFreq = 0
        if self.childTexPtnAnimMode in (TexturePatternAnimMode.Clamp, TexturePatternAnimMode.Loop):
            childTexPatFreq = self.childTexPatFreq
            assert childTexPatFreq > 0

        childNumTexPat = 1
        if self.childTexPtnAnimMode == TexturePatternAnimMode.Random:
            childNumTexPat = self.childNumTexPat
            assert 1 < childNumTexPat <= 0xFF

        isChildTexPatAnimClump = self.childTexPtnAnimMode == TexturePatternAnimMode.Clamp

        assert self.childNumTexDivX > 0
        assert self.childNumTexDivY > 0
        assert 1 <= self.childTexPatTblUse < 32

        return b''.join((
            struct.pack(
                '>4i2f',
                self.childEmitRate,
                self.childEmitTiming,
                self.childLife,
                self.childEmitStep,
                self.childVelInheritRate,
                self.childFigureVel
            ),
            struct.pack(VEC3_FMT, *self.childRandVel),
            struct.pack(F32_FMT, self.childInitPosRand),
            self.childPrimitiveFigure.save(),
            struct.pack(
                '>f4I',
                self.childDynamicsRandom,
                self.childBlendType,
                self.childMeshType,
                self.childBillboardType,
                self.childZBufATestType
            ),
            self.childTex.save(),
            struct.pack(U32_FMT, self.childDisplaySide),
            struct.pack(VEC3_FMT, *self.childColor0),
            struct.pack(VEC3_FMT, *self.childColor1),
            struct.pack(
                '>f2I4f',
                self.childColorScale,
                self.primitiveColorBlend,
                self.primitiveAlphaBlend,
                self.childAlpha,
                self.childAlphaTarget,
                self.childAlphaInit,
                self.childScaleInheritRate
            ),
            struct.pack(VEC2_FMT, *self.childScale),
            struct.pack(
                '>fI',
                self.childScaleRand,
                self.childRotType
            ),
            struct.pack(VEC3_FMT, childInitRotX,     childInitRotY,     childInitRotZ),
            struct.pack(VEC3_FMT, childInitRotRandX, childInitRotRandY, childInitRotRandZ),
            struct.pack(VEC3_FMT, childRotVelX,      childRotVelY,      childRotVelZ),
            struct.pack(VEC3_FMT, childRotVelRandX,  childRotVelRandY,  childRotVelRandZ),
            struct.pack(F32_FMT, self.childRotRegist),
            struct.pack(VEC2_FMT, *self.childRotBasis),
            struct.pack(VEC3_FMT, *self.childGravity),
            struct.pack(
                '>3i',
                self.childAlphaStartFrame,
                self.childAlphaBaseFrame,
                self.childScaleStartFrame
            ),
            struct.pack(VEC2_FMT, *self.childScaleTarget),
            struct.pack(
                '>H2B2f',
                childNumTexPat,
                self.childNumTexDivX,
                self.childNumTexDivY,
                childTexUScale,
                childTexVScale
            ),
            struct.pack(f'>{SystemConstants.TexturePatternNum}B', *self.childTexPatTbl),
            struct.pack(
                '>2hB3x2If4B4f16s16s2I',
                childTexPatFreq,
                self.childTexPatTblUse,
                int(isChildTexPatAnimClump),
                self.childCombinerType,
                AlphaCombinerType.construct(self.childAlphaBaseCombinerType, self.childAlphaCommonSource),
                self.childAirRegist,
                self.childShaderType,
                self.childUserShaderSetting,
                int(self.childShaderUseSoftEdge),
                1,
                self.childShaderParam0,
                self.childShaderParam1,
                self.childSoftFadeDistance,
                self.childSoftVolumeParam,
                childUserShaderDefine1,
                childUserShaderDefine2,
                self.childUserShaderFlag,
                self.childUserShaderSwitchFlag
            ),
            self.childUserShaderParam.save()
        ))

    def toDict(self, enable: bool) -> DictGeneric:
        return {
            "general": {
                "enable": {
                    "description": "Enable child emitters?\n" \
                                   "When enabled, each particle emitted by the original emitter becomes a new (child) emitter.",
                    "value": enable
                },
                "drawBeforeParent": {
                    "description": "Draw child particles before parent?",
                    "value": self.childDrawBeforeParent
                },
                "followEmitter": {
                    "description": "Enable perfect following of parent emitter SRT?\n" \
                                   "When enabled, child particles after emission move in accordance with parent emitter transform animations.",
                    "value": self.childEmitterFollow
                },
                "parentField": {
                    "description": "Enable the effect of parent emitter fields on child particles?",
                    "value": self.childParentField
                }
            },
            "emission": {
                "timing": {
                    "emitRate": {
                        "description": "Emission rate. This is the number of particles emitted per emission event.\n" \
                                       "Value must be a positive integer.",
                        "value": self.childEmitRate
                    },
                    "startFramePercent": {
                        "description": "Emission start frame as percentage of parent particle lifespan.\n" \
                                       "Value must be in the range `[0, 100]`.",
                        "value": self.childEmitTiming
                    },
                    "emitStep": {
                        "description": "Emission interval in frames. The number of frames between each emission event.",
                        "value": self.childEmitStep
                    }
                },
                "posAndInitVel": {
                    "initPosRand": {
                        "description": "Particle initial position randomizer sphere radius.\n" \
                                       "If nonzero, the actual initial position of the particle ends up being a random point on the surface of a sphere centered on the original initial position of the particle. This value defines the radius of that sphere.\n" \
                                       "Value must be positive.",
                        "value": self.childInitPosRand
                    },
                    "inheritInitVel": {
                        "enable": {
                            "description": "Inherit initial velocity from parent particle velocity?",
                            "value": self.childVelInherit
                        },
                        "scale": {
                            "description": "The scale of the inherited initial velocity from parent particle velocity.",
                            "value": self.childVelInheritRate
                        }
                    },
                    "figureVel": {
                        "description": "Magnitude of random-direction initial velocity.\n" \
                                       "This is the magnitude of the particle initial velocity in the direction of emission (and that direction is randomized to begin with).\n" \
                                       "(Added on top of inherited initial velocity if available.)",
                        "value": self.childFigureVel
                    },
                    "spreadVec": {
                        "description": "Initial velocity spread vector. This attribute specifies the range of random variation applied to a particle initial velocity in the X, Y, and Z directions. It consists of three components:\n" \
                                       "* A random decimal value picked from the range `[-x, x)` is added to the X component of the particle initial velocity.\n" \
                                       "* A random decimal value picked from the range `[-y, y)` is added to the Y component of the particle initial velocity.\n" \
                                       "* A random decimal value picked from the range `[-z, z)` is added to the Z component of the particle initial velocity.\n" \
                                       "Values must be positive.",
                        "value": Vec3ToDict(self.childRandVel)
                    },
                    "airRegist": {
                        "description": "Air resistance. This value is used to scale the particle velocity on every frame, and can be used to reduce or increase the particle velocity.\n" \
                                       "Value must be positive.",
                        "value": self.childAirRegist
                    }
                },
                "gravity": {
                    "vector": {
                        "description": "Gravity vector. This value is added to the velocity and can also be thought of as the acceleration vector.",
                        "value": Vec3ToDict(self.childGravity)
                    }
                }
            },
            "particle": {
                "lifespan": {
                    "description": "Child particle lifespan. The duration of the child particle life from emission until expiration, in frames.\n" \
                                   "Value must be greater than zero.",
                    "value": self.childLife
                },
                "momentumRnd": {
                    "description": "Momentum random scale range. Value must be positive.\n" \
                                   "Makes individual particle motion appear random:\n" \
                                   "A random decimal value picked from the range `(1-momentumRnd, 1+momentumRnd]` is used to scale the particle velocity (as well as the Spin, Convergence and PosAdd fields).",
                    "value": self.childDynamicsRandom
                },
                "shape": {
                    "type": {
                        "description": "Child particle mesh type. Possible types are:\n" \
                                       "Particle:  The particle is a flat sheet.\n" \
                                       "Primitive: The particle uses a primitive model.\n" \
                                       "(Unlike parent particles, child particles cannot use the `Stripe` type.)",
                        "value": self.childMeshType.name
                    },
                    "primitiveFigure": self.childPrimitiveFigure.toDict(),
                    "billboardType": {
                        "description": "Billboard (particle orientation) type. Possible types are:\n" \
                                       "Billboard:          Normal billboard. The particle Y-axis is bound to the camera `up` vector and the Z-axis is always parallel to the camera lens axis.\n" \
                                       "YBillboard:         Y-axis billboard. Displays the view after rotating the particle around the Y-axis only, so that its Z-axis is parallel to the camera lens axis.\n" \
                                       "PolygonXY:          XY-plane polygon. This type of particle has sides in the X direction and the Y direction.\n" \
                                       "PolygonXZ:          XZ-plane polygon. This type of particle has sides in the X direction and the Z direction.\n" \
                                       "VelLook:            Directional Y-billboard. As this type of particle moves, it looks at its position in the previous frame and tilts in that direction of movement. It rotates only around the Y-axis to display with the Z-axis facing the camera.\n" \
                                       "VelLookPolygon:     Directional polygon. As this type of particle moves, it looks at its position in the previous frame and tilts in that direction of movement.\n" \
                                       "(Since it is not possible for the mesh type to be `Stripe`, the `HistoricalStripe` & `ConsolidatedStripe` types cannot be used.)",
                        "value": self.childBillboardType.name
                    },
                    "pivotOffset": {
                        "description": "Sets the offset of the pivot position for particle scaling and rotation, in local coordinates.",
                        "value": Vec2ToDict(self.childRotBasis)
                    }
                },
                "renderState": {
                    "blendType": {
                        "description": "Blend type. Possible types are:\n" \
                                       "Normal: Standard Blending, blends the source and destination colors using the source alpha value for smooth transparency effects, where the source partially covers the destination.\n" \
                                       "Add:    Additive Blending, adds the source color to the destination color, creating brightening effects such as glows or light flares.\n" \
                                       "Sub:    Subtractive Blending, subtracts the source color from the destination color using reverse subtraction, often used for creating inverted or darkening effects.\n" \
                                       "Screen: Screen Blending, combines the source and destination colors by inverting, multiplying, and adding them, useful for lightening the image and creating highlights.\n" \
                                       "Mult:   Multiplicative Blending, multiplies the source color with the destination color, commonly used for tinting, shading, or creating shadow effects.",
                        "value": self.childBlendType.name
                    },
                    "zBufATestType": {
                        "description": "Z-buffer alpha test type. Possible types are:\n" \
                                       "Normal:  Translucent Rendering (with depth testing), enables depth testing and sets the depth function to allow writing to fragments that are closer or at the same depth as existing ones, disables depth writing, and enables blending for proper rendering of transparent objects.\n" \
                                       "ZIgnore: Translucent Rendering (without depth testing), disables depth testing and depth writing, and enables blending, typically used for rendering effects that do not require depth sorting.\n" \
                                       "Entity:  Opaque Rendering (with depth and alpha testing), enables depth testing with depth writing, sets the depth function to allow writing to fragments that are closer or at the same depth as existing ones, uses alpha testing to discard fragments with alpha less than or equal to 0.5, and disables blending, making it suitable for rendering fully opaque objects that need depth sorting.",
                        "value": self.childZBufATestType.name
                    },
                    "displaySide": {
                        "description": "Which side to display. Possible values are:\n" \
                                       "Both:  Display both sides of particles.\n" \
                                       "Front: Display only front side of particles.\n" \
                                       "Back:  Display only back side of particles.",
                        "value": self.childDisplaySide.name
                    }
                }
            },
            "combiner": {
                "mode": {
                    "description": "Pixel combiner mode. Possible modes are:\n" \
                                   "- Normal:\n" \
                                   "In this mode, the new pixel output is calculated as follows:\n" \
                                   "PixelColor = TextureColor;\n" \
                                   "PixelColor = PtclColorBlendFunc(PixelColor);\n" \
                                   "if (MeshTypeIsPrimitive) PixelColor = PrimitiveColorBlendFunc(PixelColor);\n" \
                                   "PixelAlpha = TextureAlpha;\n" \
                                   "if (MeshTypeIsPrimitive) PixelAlpha = PrimitiveAlphaBlendFunc(PixelAlpha);\n" \
                                   "PixelAlpha = PtclAlphaBlendFunc(PixelAlpha);\n" \
                                   "\n" \
                                   "- Refraction:\n" \
                                   "Color Buffer Refraction. Causes distortion of the background of the particle.\n" \
                                   "In this mode, the new pixel output is calculated as follows:\n" \
                                   "vec2 Offset = vec2(TextureColor.r, TextureAlpha) * offsetScale;\n" \
                                   "PixelColor = GetOriginalPixel(Offset);\n" \
                                   "PixelColor = PtclColorBlendFunc(PixelColor);\n" \
                                   "if (MeshTypeIsPrimitive) PixelColor = PrimitiveColorBlendFunc(PixelColor);\n" \
                                   "float AlphaTemp = 1.0;\n" \
                                   "if (MeshTypeIsPrimitive) AlphaTemp = PrimitiveAlphaBlendFunc(AlphaTemp);\n" \
                                   "PixelAlpha = PtclAlphaBlendFunc(AlphaTemp);\n" \
                                   "\n" \
                                   "(`Distortion` type is not usable since child emitters only have a single texture.)",
                    "value": self.childShaderType.name
                },
                "offsetScale": {
                    "description": "Used to scale the offset in `Refraction` mode. Value must be positive.",
                    "value": Vec2ToDict((self.childShaderParam0, self.childShaderParam1))
                },
                "softParticle": {
                    "description": "This effect smooths out the edges of particles when they overlap with other objects in the scene, avoiding harsh intersections with surfaces. By enabling soft particles, you can make the particle system emit particles close to opaque surfaces without causing hard edges. Instead of having a hard, sharp edge, the particles gradually become more transparent near their boundaries, creating a softer transition between the particles and the background. This helps particles look less like flat images and more like volumetric objects.",
                    "enable": {
                        "description": "Enable soft particles?",
                        "value": self.childShaderUseSoftEdge
                    },
                    "fadeDistance": {
                        "description": "Soft particle fade distance. This parameter sets how far from the point of overlap between the particle and the model the softening effect will start, i.e., the transition from opaque to transparent. A larger distance results in a more gradual fade.\n" \
                                       "Value must be positive.",
                        "value": self.childSoftFadeDistance
                    },
                    "volume": {
                        "description": "Adjusts the perceived thickness of the particles. Higher values make the particles appear thicker by influencing how transparency is calculated based on the particle color brightness.\n" \
                                       "Value must be positive.",
                        "value": self.childSoftVolumeParam
                    }
                },
                "colorCombiner": {
                    "texture": {
                        "source": {
                            "description": "`TextureColor` value source.\n" \
                                           "Possible sources are:\n" \
                                           "RGB: Color data of child texture.\n" \
                                           "One: Constant value (1.0).",
                            "value": self.childTextureColorSource.name
                        }
                    },
                    "ptclColor": {
                        "colorBlendType": {
                            "description": "Color calculation formula.\n" \
                                           "Possible types are:\n" \
                                           "- Color:\n" \
                                           "PtclColorBlendFunc(Color) = PtclColor0;\n" \
                                           "- Texture:\n" \
                                           "PtclColorBlendFunc(Color) = Color * PtclColor0;\n" \
                                           "- TextureInterpolate:\n" \
                                           "PtclColorBlendFunc(Color) = (Color * PtclColor0) + ((1 - Color) * PtclColor1);\n" \
                                           "- TextureAdd:\n" \
                                           "PtclColorBlendFunc(Color) = (Color * PtclColor0) + PtclColor1;",
                            "value": self.childCombinerType.name
                        }
                    },
                    "primitive": {
                        "source": {
                            "description": "`PrimitiveColor` value source.\n" \
                                           "Possible sources are:\n" \
                                           "RGB: Color data of primitive.\n" \
                                           "One: Constant value (1.0).",
                            "value": self.childPrimitiveColorSource.name
                        },
                        "colorBlendType": {
                            "description": "Type of color blending with `PrimitiveColor`.\n" \
                                           "Possible types are:\n" \
                                           "- Mod:\n" \
                                           "PrimitiveColorBlendFunc(Color) = Color * PrimitiveColor;\n" \
                                           "- Add:\n" \
                                           "PrimitiveColorBlendFunc(Color) = Color + PrimitiveColor;\n" \
                                           "- Sub:\n" \
                                           "PrimitiveColorBlendFunc(Color) = Color - PrimitiveColor;",
                            "value": self.primitiveColorBlend.name
                        }
                    }
                },
                "alphaCombiner": {
                    "commonSource": {
                        "description": "Common alpha source from child texture and primitive.\n" \
                                       "Possible sources are:\n" \
                                       "Alpha: Alpha channel of child texture and primitive color data.\n" \
                                       "Red:   Red channel of child texture and primitive color data.",
                        "value": self.childAlphaCommonSource.name
                    },
                    "texture": {
                        "source": {
                            "description": "`TextureAlpha` value source.\n" \
                                           "Possible sources are:\n" \
                                           "Pass: Select from `commonSource`.\n" \
                                           "One:  Constant value (1.0).",
                            "value": self.childTextureAlphaSource.name
                        }
                    },
                    "primitive": {
                        "source": {
                            "description": "`PrimitiveAlpha` value source.\n" \
                                           "Possible sources are:\n" \
                                           "Pass: Select from `commonSource`.\n" \
                                           "One:  Constant value (1.0).",
                            "value": self.childPrimitiveAlphaSource.name
                        },
                        "alphaBlendType": {
                            "description": "Type of alpha blending with `PrimitiveAlpha`.\n" \
                                           "Possible types are:\n" \
                                           "- Mod:\n" \
                                           "PrimitiveAlphaBlendFunc(Alpha) = Alpha * PrimitiveAlpha;\n" \
                                           "- Add:\n" \
                                           "PrimitiveAlphaBlendFunc(Alpha) = Alpha + PrimitiveAlpha;\n" \
                                           "- Sub:\n" \
                                           "PrimitiveAlphaBlendFunc(Alpha) = Alpha - PrimitiveAlpha;",
                            "value": self.primitiveAlphaBlend.name
                        }
                    },
                    "ptclAlpha": {
                        "alphaBlendType": {
                            "description": "Alpha calculation formula.\n" \
                                           "Possible types are:\n" \
                                           "Mod:\n" \
                                           "PtclAlphaBlendFunc(Alpha) = Alpha * PtclAlpha;\n" \
                                           "Sub:\n" \
                                           "PtclAlphaBlendFunc(Alpha) = (Alpha - (1 - PtclAlpha)) * 2;",
                            "value": self.childAlphaBaseCombinerType.name
                        }
                    }
                }
            },
            "texture": TextureToDictEx(
                self.childTex,
                self.childNumTexDivX, self.childNumTexDivY,
                self.childTexAddressingMode,
                None,
                self.childTexPtnAnimMode,
                self.childNumTexPat,
                self.isChildTexPatAnimRand,
                self.childTexPatFreq,
                self.childTexPatTblUse,
                self.childTexPatTbl
            ),
            "ptclColor": {
                "colorScale": {
                    "description": "Color scale. Used to scale the final color 0 & color 1 values.\n" \
                                   "Value must be positive.",
                    "value": self.childColorScale
                },
                **{f"color{i}": {
                    "inherit": {
                        "description": f"Inherit color {i} from parent particle color?",
                        "value": getattr(self, f"childColor{i}Inherit")
                    },
                    "constant": {
                        "description": f"If not inhertied, set color {i} to this constant value.",
                        "value": Color3ToDict(getattr(self, f"childColor{i}"))
                    }
                } for i in range(ColorKind.Max)}
            },
            "ptclAlpha": {
                "description": "The value of the particle alpha attribute is animated using 3 alpha elements in 3 sections that are fitted to the particle lifespan. If `alphaSection1` is set to 0 AND `alphaSection2` is set to 100, the alpha animation is disabled.",
                "animationParam": {
                    "alphaElem[0]": {
                        "description": "Alpha animation element 0.\n" \
                                       "Value must be in the range `[0, 1]`.",
                        "value": self.childAlphaInit
                    },
                    "alphaElem[1]": {
                        "description": "Alpha animation element 1.\n" \
                                       "Value must be in the range `[0, 1]`.",
                        "inherit": {
                            "description": "Inherit value from parent particle alpha?",
                            "value": self.childAlphaInherit
                        },
                        "value": self.childAlpha
                    },
                    "alphaElem[2]": {
                        "description": "Alpha animation element 2.\n" \
                                       "Value must be in the range `[0, 1]`.",
                        "value": self.childAlphaTarget
                    },
                    "alphaSection1": {
                        "description": "End of 1st alpha section since emission, in frames.\n" \
                                       "Value must be in the range `[0, lifespan]`.\n" \
                                       "During this section, the alpha value transitions from `alphaElem[0]` to `alphaElem[1]` (using linear interpolation).",
                        "value": self.childAlphaBaseFrame
                    },
                    "alphaSection2": {
                        "description": "End of 2nd alpha section since emission, in frames.\n" \
                                       "Value must be in the range `[0, lifespan]`.\n" \
                                       "During this section, the alpha value is fixed to `alphaElem[1]`.\n" \
                                       "(In the 3rd alpha section (which is last), the alpha value transitions from `alphaElem[1]` to `alphaElem[2]` (using linear interpolation).)",
                        "value": self.childAlphaStartFrame
                    }
                }
            },
            "ptclScale": {
                "description": "The value of the particle scale attribute is animated by adding a fixed total value over a certain period of the particle lifespan.\n" \
                               "Once a certain number of frames passes, the particle scale is increased on every frame by a fixed rate that is calculated as `((target - 1) / (lifespan - startFrame)) * baseScale`.",
                "baseScale": {
                    "description": "Particle Scale base value. Must be positive.",
                    "inherit": {
                        "enable": {
                            "description": "Inherit value from parent particle scale?",
                            "value": self.childScaleInherit
                        },
                        "scale": {
                            "description": "The scale of the inherited value from parent particle scale. Value must be positive.",
                            "value": self.childScaleInheritRate
                        }
                    },
                    "value": Vec2ToDict(self.childScale)
                },
                "baseScaleRand": {
                    "description": "Particle Scale base value random percentage range.\n" \
                                   "A random decimal value picked from the range `(1-baseScaleRand, 1]` is used to scale the particle scale base value (`baseScale`).\n" \
                                   "Value must be in the range `[0, 1]`, where a value of 1 is equivalent to 100%.",
                    "value": self.childScaleRand
                },
                "animationParam": {
                    "startFrame": {
                        "description": "Start frame (since emission) of addition animation. Prior to the specified frame, the particle scale is fixed to `baseScale`.\n" \
                                       "Value must be in the range `[0, lifespan]`.",
                        "value": self.childScaleStartFrame
                    },
                    "target": {
                        "description": "Animation target. This value is used in calculating the fixed rate that is added to the particle scale on every frame during the animation.\n" \
                                       "Value must be positive.",
                        "value": Vec2ToDict(self.childScaleTarget)
                    }
                }
            },
            "ptclRot": {
                "type": {
                    "description": "Particle rotation type. Possible types are:\n" \
                                   "NoWork: Particles do not rotate.\n" \
                                   "RotX:   Particles can rotate only around on the X axis.\n" \
                                   "RotY:   Particles can rotate only around on the Y axis.\n" \
                                   "RotZ:   Particles can rotate only around on the Z axis.\n" \
                                   "RotXYZ: Particles can rotate around all axes.",
                    "value": self.childRotType.name
                },
                "initRot": {
                    "description": "Particle initial rotation, in radians.\n" \
                                  f"Value must be in the range `[0, {MATH_PI_2}]`.",
                    "inherit": {
                        "description": "Inherit value from parent particle rotation?",
                        "value": self.childRotInherit
                    },
                    "value": Vec3ToDict(self.childInitRot)
                },
                "initRotRand": {
                    "description": "Particle initial rotation random range size, in radians.\n" \
                                   "A random decimal value picked from the range `[0, initRotRand)` is added to `initRot`.\n" \
                                  f"Value must be in the range `[0, {MATH_PI_2}]`.",
                    "value": Vec3ToDict(self.childInitRotRand)
                },
                "rotRegist": {
                    "description": "Particle rotation air resistance. This value is used to scale the particle angular velocity on every frame, and can be used to reduce or increase the particle angular velocity.\n" \
                                   "Value must be positive.",
                    "value": self.childRotRegist
                },
                "rotVel": {
                    "description": "Particle angular velocity, in radians.\n" \
                                  f"Value must be in the range `[{-MATH_PI}, {MATH_PI}]`.",
                    "value": Vec3ToDict(self.childRotVel)
                },
                "rotVelRand": {
                    "description": "Particle angular velocity random range size, in radians.\n" \
                                   "A random decimal value picked from the range `[0, rotVelRand)` is added to `rotVel`.\n" \
                                  f"Value must be in the range `[0, {MATH_PI_2}]`.",
                    "value": Vec3ToDict(self.childRotVelRand)
                }
            },
            "userShader": {
                "shaderType": {
                    "description": "Games and applications are allowed to select a global shader type (ref. `nw::eft::Renderer::SetShaderType` function) that define special environments under which user shaders are allowed to behave differently. Three global types exist (0, 1, 2). Type 0 is the type under which user shaders should behave as normal. Types 1 and 2 are special types that user shaders can react to.",
                    "macroDef1": {
                        "description": "A macro to be dynamically defined at shader compilation when the global shader type is set to 1. The user shader can detect if the global type is set to 1 by checking if this macro has been defined. Value must be encodeable to less than 16 bytes using Shift JIS.",
                        "value": self.childUserShaderDefine1 if self.childUserShaderDefine1 else None
                    },
                    "macroDef2": {
                        "description": "A macro to be dynamically defined at shader compilation when the global shader type is set to 2. The user shader can detect if the global type is set to 2 by checking if this macro has been defined. Value must be encodeable to less than 16 bytes using Shift JIS.",
                        "value": self.childUserShaderDefine2 if self.childUserShaderDefine2 else None
                    }
                },
                "localType": {
                    "description": "Local shader type that can be selected per emitter, in contrast to the global shader type. Nine local types exist (0-8). Type 0 is the type under which user shaders should behave as normal. Types 1 to 8 are special types that user shaders can react to.\n" \
                                   "- If the value is set to 0, the macros `USR_SETTING_NONE`, `USR_VERTEX_SETTING_NONE` and `USR_FRAGMENT_SETTING_NONE` are dynamically defined at shader compilation.\n" \
                                   "- If the value is set to some number `X` between 1 and 8, the macros `USR_SETTING_X`, `USR_VERTEX_SETTING_X` and `USR_FRAGMENT_SETTING_X` are dynamically defined at shader compilation.",
                    "value": self.childUserShaderSetting
                },
                "bitfield": {
                    "description": "A 32-bit bitfield specifying 32 flags which are possible to combine.\n" \
                                   "For each bit X between 0 and 31 that is set to 1, the macros `USR_FLAG_X`, `USR_VERTEX_FLAG_X` and `USR_FRAGMENT_FLAG_X` are dynamically defined at shader compilation.",
                    "value": f'0b{self.childUserShaderFlag:032b}'
                },
                "switchCase": {
                    "description": "A 32-bit bitfield specifying 32 switch cases.\n" \
                                   "For each bit X between 0 and 31 that is set to 1, the macros `USR_SWITCH_FLAG_X`, `USR_VERTEX_SWITCH_FLAG_X` and `USR_FRAGMENT_SWITCH_FLAG_X` are dynamically defined at shader compilation.",
                    "value": f'0b{self.childUserShaderSwitchFlag:032b}'
                },
                "param": self.childUserShaderParam.toDict()
            },
            "unused": {
                "description": "Attributes which are stored in the child emitter data, but not actually used in the code, and therefore they are useless.",
                "isWorldField": {
                    "value": self.childWorldField
                }
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        dic_general: DictGeneric = dic["general"]
        dic_emission: DictGeneric = dic["emission"]
        dic_emission_timing: DictGeneric = dic_emission["timing"]
        dic_emission_posAndInitVel: DictGeneric = dic_emission["posAndInitVel"]
        dic_emission_posAndInitVel_inheritInitVel: DictGeneric = dic_emission_posAndInitVel["inheritInitVel"]
        dic_particle: DictGeneric = dic["particle"]
        dic_particle_shape: DictGeneric = dic_particle["shape"]
        dic_particle_renderState: DictGeneric = dic_particle["renderState"]
        dic_combiner: DictGeneric = dic["combiner"]
        dic_combiner_softParticle: DictGeneric = dic_combiner["softParticle"]
        dic_combiner_colorCombiner: DictGeneric = dic_combiner["colorCombiner"]
        dic_combiner_colorCombiner_primitive: DictGeneric = dic_combiner_colorCombiner["primitive"]
        dic_combiner_alphaCombiner: DictGeneric = dic_combiner["alphaCombiner"]
        dic_combiner_alphaCombiner_primitive: DictGeneric = dic_combiner_alphaCombiner["primitive"]
        dic_ptclColor: DictGeneric = dic["ptclColor"]
        dic_ptclAlpha_animationParam: DictGeneric = dic["ptclAlpha"]["animationParam"]
        dic_ptclAlpha_animationParam_alphaElem1: DictGeneric = dic_ptclAlpha_animationParam["alphaElem[1]"]
        dic_ptclScale: DictGeneric = dic["ptclScale"]
        dic_ptclScale_baseScale: DictGeneric = dic_ptclScale["baseScale"]
        dic_ptclScale_baseScale_inherit: DictGeneric = dic_ptclScale_baseScale["inherit"]
        dic_ptclScale_animationParam: DictGeneric = dic_ptclScale["animationParam"]
        dic_ptclRot: DictGeneric = dic["ptclRot"]
        dic_ptclRot_initRot: DictGeneric = dic_ptclRot["initRot"]
        dic_userShader: DictGeneric = dic["userShader"]
        dic_userShader_shaderType: DictGeneric = dic_userShader["shaderType"]

        enable = VerifyBool(dic_general["enable"]["value"])
        self.childDrawBeforeParent = VerifyBool(dic_general["drawBeforeParent"]["value"])
        self.childEmitterFollow = VerifyBool(dic_general["followEmitter"]["value"])
        self.childParentField = VerifyBool(dic_general["parentField"]["value"])

        self.childEmitRate = VerifyS32Positive(dic_emission_timing["emitRate"]["value"])
        self.childEmitTiming = VerifyIntRange(dic_emission_timing["startFramePercent"]["value"], 0, 100)
        self.childEmitStep = VerifyS32Positive(dic_emission_timing["emitStep"]["value"])
        self.childInitPosRand = VerifyF32Positive(dic_emission_posAndInitVel["initPosRand"]["value"])
        self.childVelInherit = VerifyBool(dic_emission_posAndInitVel_inheritInitVel["enable"]["value"])
        self.childVelInheritRate = VerifyF32(dic_emission_posAndInitVel_inheritInitVel["scale"]["value"])
        self.childFigureVel = VerifyF32(dic_emission_posAndInitVel["figureVel"]["value"])
        self.childRandVel = Vec3PositiveFromDict(dic_emission_posAndInitVel["spreadVec"]["value"])
        self.childAirRegist = VerifyF32Positive(dic_emission_posAndInitVel["airRegist"]["value"])
        self.childGravity = Vec3FromDict(dic_emission["gravity"]["vector"]["value"])

        self.childLife = VerifyS32PositiveNonZero(dic_particle["lifespan"]["value"])
        self.childDynamicsRandom = VerifyF32Positive(dic_particle["momentumRnd"]["value"])
        self.childMeshType = MeshType[dic_particle_shape["type"]["value"]]
        self.childPrimitiveFigure.fromDict(dic_particle_shape["primitiveFigure"])
        self.childBillboardType = BillboardType[dic_particle_shape["billboardType"]["value"]]
        self.childRotBasis = Vec2FromDict(dic_particle_shape["pivotOffset"]["value"])
        self.childBlendType = BlendType[dic_particle_renderState["blendType"]["value"]]
        self.childZBufATestType = ZBufATestType[dic_particle_renderState["zBufATestType"]["value"]]
        self.childDisplaySide = DisplaySideType[dic_particle_renderState["displaySide"]["value"]]

        if self.childMeshType == MeshType.Primitive:
            assert self.childPrimitiveFigure.index != 0xFFFFFFFF
        else:
            assert self.childPrimitiveFigure.index == 0xFFFFFFFF
            assert self.childMeshType == MeshType.Particle

        assert self.childBillboardType not in (BillboardType.Primitive, BillboardType.HistoricalStripe, BillboardType.ConsolidatedStripe)

        self.childShaderType = FragmentShaderVariation[dic_combiner["mode"]["value"]]
        self.childShaderParam0, self.childShaderParam1 = Vec2PositiveFromDict(dic_combiner["offsetScale"]["value"])
        self.childShaderUseSoftEdge = VerifyBool(dic_combiner_softParticle["enable"]["value"])
        self.childSoftFadeDistance = VerifyF32Positive(dic_combiner_softParticle["fadeDistance"]["value"])
        self.childSoftVolumeParam = VerifyF32Positive(dic_combiner_softParticle["volume"]["value"])
        self.childTextureColorSource = ColorSource[dic_combiner_colorCombiner["texture"]["source"]["value"]]
        self.childCombinerType = ColorCombinerType[dic_combiner_colorCombiner["ptclColor"]["colorBlendType"]["value"]]
        self.childPrimitiveColorSource = ColorSource[dic_combiner_colorCombiner_primitive["source"]["value"]]
        self.primitiveColorBlend = ColorAlphaBlendType[dic_combiner_colorCombiner_primitive["colorBlendType"]["value"]]
        self.childAlphaCommonSource = AlphaCommonSource[dic_combiner_alphaCombiner["commonSource"]["value"]]
        self.childTextureAlphaSource = AlphaSource[dic_combiner_alphaCombiner["texture"]["source"]["value"]]
        self.childPrimitiveAlphaSource = AlphaSource[dic_combiner_alphaCombiner_primitive["source"]["value"]]
        self.primitiveAlphaBlend = ColorAlphaBlendType[dic_combiner_alphaCombiner_primitive["alphaBlendType"]["value"]]
        self.childAlphaBaseCombinerType = AlphaBaseCombinerType[dic_combiner_alphaCombiner["ptclAlpha"]["alphaBlendType"]["value"]]

        assert self.childShaderType != FragmentShaderVariation.Distortion

        (
            self.childNumTexDivX, self.childNumTexDivY,
            self.childTexAddressingMode,
            uvDict,
            self.childTexPtnAnimMode,
            self.childNumTexPat,
            self.isChildTexPatAnimRand,
            self.childTexPatFreq,
            self.childTexPatTblUse,
            self.childTexPatTbl
        ) = TextureFromDictEx(self.childTex, dic["texture"])
        assert uvDict is None

        self.childColorScale = VerifyF32Positive(dic_ptclColor["colorScale"]["value"])
        for i in range(ColorKind.Max):
            dic_ptclColor_color = dic_ptclColor[f"color{i}"]
            setattr(self, f"childColor{i}Inherit", VerifyBool(dic_ptclColor_color["inherit"]["value"]))
            setattr(self, f"childColor{i}", Color3PositiveFromDict(dic_ptclColor_color["constant"]["value"]))

        self.childAlphaInit = VerifyF32Normal(dic_ptclAlpha_animationParam["alphaElem[0]"]["value"])
        self.childAlphaInherit = VerifyBool(dic_ptclAlpha_animationParam_alphaElem1["inherit"]["value"])
        self.childAlpha = VerifyF32Normal(dic_ptclAlpha_animationParam_alphaElem1["value"])
        self.childAlphaTarget = VerifyF32Normal(dic_ptclAlpha_animationParam["alphaElem[2]"]["value"])
        self.childAlphaBaseFrame = VerifyIntRange(dic_ptclAlpha_animationParam["alphaSection1"]["value"], 0, self.childLife)
        self.childAlphaStartFrame = VerifyIntRange(dic_ptclAlpha_animationParam["alphaSection2"]["value"], 0, self.childLife)

        self.childScaleInherit = VerifyBool(dic_ptclScale_baseScale_inherit["enable"]["value"])
        self.childScaleInheritRate = VerifyF32Positive(dic_ptclScale_baseScale_inherit["scale"]["value"])
        self.childScale = Vec2PositiveFromDict(dic_ptclScale_baseScale["value"])
        self.childScaleRand = VerifyF32Normal(dic_ptclScale["baseScaleRand"]["value"])
        self.childScaleStartFrame = VerifyIntRange(dic_ptclScale_animationParam["startFrame"]["value"], 0, self.childLife)
        self.childScaleTarget = Vec2PositiveFromDict(dic_ptclScale_animationParam["target"]["value"])

        self.childRotType = PtclRotType[dic_ptclRot["type"]["value"]]
        self.childRotInherit = VerifyBool(dic_ptclRot_initRot["inherit"]["value"])
        (childInitRotX,     childInitRotY,     childInitRotZ)     = Vec3FromDict(dic_ptclRot_initRot["value"])
        (childInitRotRandX, childInitRotRandY, childInitRotRandZ) = Vec3FromDict(dic_ptclRot["initRotRand"]["value"])
        self.childRotRegist = VerifyF32Positive(dic_ptclRot["rotRegist"]["value"])
        (childRotVelX,      childRotVelY,      childRotVelZ)      = Vec3FromDict(dic_ptclRot["rotVel"]["value"])
        (childRotVelRandX,  childRotVelRandY,  childRotVelRandZ)  = Vec3FromDict(dic_ptclRot["rotVelRand"]["value"])

        if self.childRotType == PtclRotType.NoWork:
            assert childInitRotX     == 0.0 and childInitRotY     == 0.0 and childInitRotZ     == 0.0
            assert childInitRotRandX == 0.0 and childInitRotRandY == 0.0 and childInitRotRandZ == 0.0
            assert childRotVelX      == 0.0 and childRotVelY      == 0.0 and childRotVelZ      == 0.0
            assert childRotVelRandX  == 0.0 and childRotVelRandY  == 0.0 and childRotVelRandZ  == 0.0

        elif self.childRotType == PtclRotType.RotX:
            assert childInitRotY     == 0.0 and childInitRotZ     == 0.0
            assert childInitRotRandY == 0.0 and childInitRotRandZ == 0.0
            assert childRotVelY      == 0.0 and childRotVelZ      == 0.0
            assert childRotVelRandY  == 0.0 and childRotVelRandZ  == 0.0

        elif self.childRotType == PtclRotType.RotY:
            assert childInitRotX     == 0.0 and childInitRotZ     == 0.0
            assert childInitRotRandX == 0.0 and childInitRotRandZ == 0.0
            assert childRotVelX      == 0.0 and childRotVelZ      == 0.0
            assert childRotVelRandX  == 0.0 and childRotVelRandZ  == 0.0

        elif self.childRotType == PtclRotType.RotZ:
            assert childInitRotX     == 0.0 and childInitRotY     == 0.0
            assert childInitRotRandX == 0.0 and childInitRotRandY == 0.0
            assert childRotVelX      == 0.0 and childRotVelY      == 0.0
            assert childRotVelRandX  == 0.0 and childRotVelRandY  == 0.0

        if self.childRotType in (PtclRotType.RotX, PtclRotType.RotXYZ):
            assert 0.0 <= childInitRotX <= MATH_PI_2
            assert 0.0 <= childInitRotRandX <= MATH_PI_2
            assert -MATH_PI <= childRotVelX <= MATH_PI
            assert 0.0 <= childRotVelRandX <= MATH_PI_2

        if self.childRotType in (PtclRotType.RotY, PtclRotType.RotXYZ):
            assert 0.0 <= childInitRotY <= MATH_PI_2
            assert 0.0 <= childInitRotRandY <= MATH_PI_2
            assert -MATH_PI <= childRotVelY <= MATH_PI
            assert 0.0 <= childRotVelRandY <= MATH_PI_2

        if self.childRotType in (PtclRotType.RotZ, PtclRotType.RotXYZ):
            assert 0.0 <= childInitRotZ <= MATH_PI_2
            assert 0.0 <= childInitRotRandZ <= MATH_PI_2
            assert -MATH_PI <= childRotVelZ <= MATH_PI
            assert 0.0 <= childRotVelRandZ <= MATH_PI_2

        self.childInitRot     = (childInitRotX,     childInitRotY,     childInitRotZ)
        self.childInitRotRand = (childInitRotRandX, childInitRotRandY, childInitRotRandZ)
        self.childRotVel      = (childRotVelX,      childRotVelY,      childRotVelZ)
        self.childRotVelRand  = (childRotVelRandX,  childRotVelRandY,  childRotVelRandZ)

        self.childUserShaderDefine1 = VerifyNullableStr(dic_userShader_shaderType["macroDef1"]["value"])
        self.childUserShaderDefine2 = VerifyNullableStr(dic_userShader_shaderType["macroDef2"]["value"])
        self.childUserShaderSetting = VerifyIntRange(dic_userShader["localType"]["value"], 0, 8)
        self.childUserShaderFlag = VerifyU32(int(dic_userShader["bitfield"]["value"], 2))
        self.childUserShaderSwitchFlag = VerifyU32(int(dic_userShader["switchCase"]["value"], 2))
        self.childUserShaderParam.fromDict(dic_userShader["param"])

        self.childWorldField = VerifyBool(dic["unused"]["isWorldField"]["value"])

        return enable


DefaultChildData = nw__eft__ChildData()
DefaultChildData.childColor0Inherit = False
DefaultChildData.childAlphaInherit = True
DefaultChildData.childScaleInherit = False
DefaultChildData.childRotInherit = True
DefaultChildData.childVelInherit = False
DefaultChildData.childEmitterFollow = True
DefaultChildData.childWorldField = False
DefaultChildData.childParentField = False
DefaultChildData.childDrawBeforeParent = False
DefaultChildData.isChildTexPatAnimRand = False
DefaultChildData.childColor1Inherit = False
DefaultChildData.childTextureColorSource = ColorSource.RGB
DefaultChildData.childPrimitiveColorSource = ColorSource.RGB
DefaultChildData.childTextureAlphaSource = AlphaSource.Pass
DefaultChildData.childPrimitiveAlphaSource = AlphaSource.Pass
DefaultChildData.childEmitRate = 1
DefaultChildData.childEmitTiming = 60
DefaultChildData.childLife = 60
DefaultChildData.childEmitStep = 10
DefaultChildData.childVelInheritRate = 1.0
DefaultChildData.childFigureVel = 0.1
DefaultChildData.childRandVel = (0.0, 0.0, 0.0)
DefaultChildData.childInitPosRand = 0.0
DefaultChildData.childPrimitiveFigure.dataSize = 0
DefaultChildData.childPrimitiveFigure.index = 0xFFFFFFFF
DefaultChildData.childDynamicsRandom = 0.0
DefaultChildData.childBlendType = BlendType.Normal
DefaultChildData.childMeshType = MeshType.Particle
DefaultChildData.childBillboardType = BillboardType.Billboard
DefaultChildData.childZBufATestType = ZBufATestType.Normal
DefaultChildData.childDisplaySide = DisplaySideType.Both
# DefaultChildData.childTex.width = 0
# DefaultChildData.childTex.height = 0
# DefaultChildData.childTex.tileMode = GX2TileMode(0)
# DefaultChildData.childTex.swizzle = 0
# DefaultChildData.childTex.alignment = 0
# DefaultChildData.childTex.pitch = 0
# DefaultChildData.childTex.wrapModeU = TextureWrapMode.Mirror
# DefaultChildData.childTex.wrapModeV = TextureWrapMode.Mirror
# DefaultChildData.childTex.filterMode = TextureFilterMode.Linear
# DefaultChildData.childTex.mipLevel = 0
# DefaultChildData.childTex.compSel = 0
# DefaultChildData.childTex.mipOffset = (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
# DefaultChildData.childTex.enableMipLevel = ENABLE_MIP_LEVEL_MAX
# DefaultChildData.childTex.mipMapBias = 0.0
# DefaultChildData.childTex.originalDataFormat = TextureFormat.Invalid
# DefaultChildData.childTex.originalDataPos = 0
# DefaultChildData.childTex.originalDataSize = 0
# DefaultChildData.childTex.nativeDataFormat = TextureFormat.Invalid
# DefaultChildData.childTex.nativeDataSize = 0
# DefaultChildData.childTex.nativeDataPos = 0
# DefaultChildData.childTex.imageIndex = -1
# DefaultChildData.childTex.gx2TextureIndex = -1
DefaultChildData.childColor0 = (1.0, 1.0, 1.0)
DefaultChildData.childColor1 = (1.0, 1.0, 1.0)
DefaultChildData.childColorScale = 1.0
DefaultChildData.primitiveColorBlend = ColorAlphaBlendType.Mod
DefaultChildData.primitiveAlphaBlend = ColorAlphaBlendType.Mod
DefaultChildData.childAlpha = 1.0
DefaultChildData.childAlphaTarget = 0.0
DefaultChildData.childAlphaInit = 0.0
DefaultChildData.childScaleInheritRate = 0.0
DefaultChildData.childScale = (1.0, 1.0)
DefaultChildData.childScaleRand = 0.0
DefaultChildData.childRotType = PtclRotType.NoWork
DefaultChildData.childInitRot = (0.0, 0.0, 0.0)
DefaultChildData.childInitRotRand = (0.0, 0.0, 0.0)
DefaultChildData.childRotVel = (0.0, 0.0, 0.0)
DefaultChildData.childRotVelRand = (0.0, 0.0, 0.0)
DefaultChildData.childRotRegist = 1.0
DefaultChildData.childRotBasis = (0.0, 0.0)
DefaultChildData.childGravity = (0.0, -0.0, 0.0)
DefaultChildData.childAlphaStartFrame = DefaultChildData.childLife
DefaultChildData.childAlphaBaseFrame = 0
DefaultChildData.childScaleStartFrame = DefaultChildData.childLife
DefaultChildData.childScaleTarget = (0.0, 0.0)
DefaultChildData.childNumTexPat = 1
DefaultChildData.childNumTexDivX = 1
DefaultChildData.childNumTexDivY = 1
DefaultChildData.childTexAddressingMode = TextureAddressing.U1V1
DefaultChildData.childTexPtnAnimMode = TexturePatternAnimMode.Null
DefaultChildData.childTexPatTbl = (0,) * SystemConstants.TexturePatternNum
DefaultChildData.childTexPatFreq = 1
DefaultChildData.childTexPatTblUse = 2
DefaultChildData.childCombinerType = ColorCombinerType.Texture
DefaultChildData.childAlphaBaseCombinerType = AlphaBaseCombinerType.Mod
DefaultChildData.childAlphaCommonSource = AlphaCommonSource.Alpha
DefaultChildData.childAirRegist = 1.0
DefaultChildData.childShaderType = FragmentShaderVariation.Normal
DefaultChildData.childUserShaderSetting = 0
DefaultChildData.childShaderUseSoftEdge = False
DefaultChildData.childShaderParam0 = 0.0
DefaultChildData.childShaderParam1 = 0.0
DefaultChildData.childSoftFadeDistance = 0.0
DefaultChildData.childSoftVolumeParam = 0.0
DefaultChildData.childUserShaderDefine1 = ''
DefaultChildData.childUserShaderDefine2 = ''
DefaultChildData.childUserShaderFlag = 0
DefaultChildData.childUserShaderSwitchFlag = 0
DefaultChildData.childUserShaderParam.param = (0.0,) * 32


class nw__eft__FieldRandomData:
    structSize = S32_SIZE + VEC3_SIZE
    assert structSize == 0x10

    fieldRandomBlank: int
    fieldRandomVelAdd: Vec3Numeric

    def load(self, data: ByteString, pos: int = 0) -> None:
        # When to apply randomization
        self.fieldRandomBlank = struct.unpack_from(S32_FMT, data, pos)[0]; pos += S32_SIZE
        assert self.fieldRandomBlank > 0

        # Velocity addition value
        self.fieldRandomVelAdd = struct.unpack_from(VEC3_FMT, data, pos)
        assert self.fieldRandomVelAdd[0] >= 0.0 and \
               self.fieldRandomVelAdd[1] >= 0.0 and \
               self.fieldRandomVelAdd[2] >= 0.0

    def save(self) -> bytes:
        return b''.join((
            struct.pack(S32_FMT, self.fieldRandomBlank),
            struct.pack(VEC3_FMT, *self.fieldRandomVelAdd)
        ))

    def toDict(self, enable: bool) -> DictGeneric:
        return {
            "enable": {
                "description": "Enable velocity randomization field?",
                "value": enable
            },
            "interval": {
                "description": "The time interval (in frames) between each time the field effect should take place.\n" \
                               "Must be greater than zero.",
                "value": self.fieldRandomBlank
            },
            "rnd": {
                "description": "This attribute specifies the range of random variation applied to the particle velocity, at each time interval specified by `interval`, in the X, Y, and Z directions. It consists of three components:\n" \
                                "* A random decimal value picked from the range `[-x, x)` is added to the X component of the particle velocity.\n" \
                                "* A random decimal value picked from the range `[-y, y)` is added to the Y component of the particle velocity.\n" \
                                "* A random decimal value picked from the range `[-z, z)` is added to the Z component of the particle velocity.\n" \
                                "Value must be positive.",
                "value": Vec3ToDict(self.fieldRandomVelAdd)
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        enable = VerifyBool(dic["enable"]["value"])
        self.fieldRandomBlank = VerifyS32PositiveNonZero(dic["interval"]["value"])
        self.fieldRandomVelAdd = Vec3PositiveFromDict(dic["rnd"]["value"])

        return enable


DefaultFieldRandomData = nw__eft__FieldRandomData()
DefaultFieldRandomData.fieldRandomBlank = 1
DefaultFieldRandomData.fieldRandomVelAdd = (0.0, 0.0, 0.0)


class nw__eft__FieldMagnetData:
    structSize = F32_SIZE + VEC3_SIZE + U32_SIZE
    assert structSize == 0x14

    fieldMagnetPower: Numeric
    fieldMagnetPos: Vec3Numeric
    fieldMagnetAxisX: bool
    fieldMagnetAxisY: bool
    fieldMagnetAxisZ: bool

    def load(self, data: ByteString, pos: int = 0) -> None:
        # Magnetism power
        self.fieldMagnetPower = struct.unpack_from(F32_FMT, data, pos)[0]; pos += S32_SIZE

        # Magnet position
        self.fieldMagnetPos = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Flags
        fieldMagnetFlg = struct.unpack_from(U32_FMT, data, pos)[0]
        assert (fieldMagnetFlg & 0b11111111111111111111111111111000) == 0  # Ununsed bits
        self.fieldMagnetAxisX = fieldMagnetFlg & FieldMagnetFlg.TargetX  # Target X axis
        self.fieldMagnetAxisY = fieldMagnetFlg & FieldMagnetFlg.TargetY  # Target Y axis
        self.fieldMagnetAxisZ = fieldMagnetFlg & FieldMagnetFlg.TargetZ  # Target Z axis

    def save(self) -> bytes:
        return b''.join((
            struct.pack(F32_FMT, self.fieldMagnetPower),
            struct.pack(VEC3_FMT, *self.fieldMagnetPos),
            struct.pack(U32_FMT, self.fieldMagnetAxisX * FieldMagnetFlg.TargetX |
                                 self.fieldMagnetAxisY * FieldMagnetFlg.TargetY |
                                 self.fieldMagnetAxisZ * FieldMagnetFlg.TargetZ)
        ))

    def toDict(self, enable: bool) -> DictGeneric:
        return {
            "enable": {
                "description": "Enable magnetic field?",
                "value": enable
            },
            "power": {
                "description": "Specifies the strength of the magnet, which affects the velocity of nearby particles.\n" \
                               "If a positive value is set, particles are attracted to the magnet.\n" \
                               "If a negative value is set, particles are repelled away from the magnet.",
                "value": self.fieldMagnetPower
            },
            "pos": {
                "description": "The position of the magnet, in the local space of the emitter.",
                "value": Vec3ToDict(self.fieldMagnetPos)
            },
            "targetX": {
                "description": "Is the magnet effective on the X axis?",
                "value": self.fieldMagnetAxisX
            },
            "targetY": {
                "description": "Is the magnet effective on the Y axis?",
                "value": self.fieldMagnetAxisY
            },
            "targetZ": {
                "description": "Is the magnet effective on the Z axis?",
                "value": self.fieldMagnetAxisZ
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        enable = VerifyBool(dic["enable"]["value"])
        self.fieldMagnetPower = VerifyF32(dic["power"]["value"])
        self.fieldMagnetPos = Vec3FromDict(dic["pos"]["value"])
        self.fieldMagnetAxisX = VerifyBool(dic["targetX"]["value"])
        self.fieldMagnetAxisY = VerifyBool(dic["targetY"]["value"])
        self.fieldMagnetAxisZ = VerifyBool(dic["targetZ"]["value"])

        return enable


DefaultFieldMagnetData = nw__eft__FieldMagnetData()
DefaultFieldMagnetData.fieldMagnetPower = 0.0
DefaultFieldMagnetData.fieldMagnetPos = (0.0, 0.0, 0.0)
DefaultFieldMagnetData.fieldMagnetAxisX = True
DefaultFieldMagnetData.fieldMagnetAxisY = True
DefaultFieldMagnetData.fieldMagnetAxisZ = True


class nw__eft__FieldSpinData:
    structSize = struct.calcsize('>2if')
    assert structSize == 0xC

    fieldSpinRotate: int
    fieldSpinAxis: FieldSpinAxis
    fieldSpinOuter: Numeric

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            self.fieldSpinRotate,   # Rotational force
            fieldSpinAxis,          # Axis
            self.fieldSpinOuter     # Outward velocity
        ) = struct.unpack_from('>2if', data, pos); pos += struct.calcsize('>2if')

        self.fieldSpinAxis = FieldSpinAxis(fieldSpinAxis)

    def save(self) -> bytes:
        return struct.pack(
            '>2if',
            self.fieldSpinRotate,
            self.fieldSpinAxis,
            self.fieldSpinOuter
        )

    def toDict(self, enable: bool) -> DictGeneric:
        return {
            "enable": {
                "description": "Enable spin field?",
                "value": enable
            },
            "rotVel": {
                "description": "The angular velocity around the specified axis, in SIGNED 32-bit Binary Angular Measurement (BAM) where (+/-)0x40000000 = (+/-)90 degrees.\n" \
                               "(This value is affected by the emitter `momentumRnd` attribute.)",
                "value": S32ToHexString(self.fieldSpinRotate)
            },
            "axis": {
                "description": "The axis of rotation. Possible values are `X`, `Y` and `Z`.",
                "value": self.fieldSpinAxis.name
            },
            "diffusionVel": {
                "description": "Magnitude of drift on plane perpendicular to the specified axis. This is the magnitude of an additional velocity component added to the particle position, perpendicular to the specified axis. It causes the particle to drift further in a direction influenced by its current position on the plane perpendicular to the specified axis.\n" \
                               "(This value is affected by the emitter `momentumRnd` attribute.)",
                "value": self.fieldSpinOuter
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        enable = VerifyBool(dic["enable"]["value"])
        self.fieldSpinRotate = VerifyS32(int(dic["rotVel"]["value"], 16))
        self.fieldSpinAxis = FieldSpinAxis[dic["axis"]["value"]]
        self.fieldSpinOuter = VerifyF32(dic["diffusionVel"]["value"])

        return enable


DefaultFieldSpinData = nw__eft__FieldSpinData()
DefaultFieldSpinData.fieldSpinRotate = 0
DefaultFieldSpinData.fieldSpinAxis = FieldSpinAxis.Y
DefaultFieldSpinData.fieldSpinOuter = 0.0


class nw__eft__FieldCollisionData:
    structSize = struct.calcsize('>2H2f')
    assert structSize == 0xC

    fieldCollisionType: FieldCollisionReaction
    fieldCollisionIsWorld: bool
    fieldCollisionCoord: Numeric
    fieldCollisionCoef: Numeric

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            fieldCollisionType,         # Type (0: Cessation (particle stops at collision surface and stays alive for only 1 more frame), 1: Reflection)
            fieldCollisionIsWorld,      # Should collision be processed in world-space (global coordinates) or local-space (relative to the particle)?
            self.fieldCollisionCoord,   # Y coordinate of the collision surface the particle interacts with
            self.fieldCollisionCoef     # Coefficient of reflection upon collision
        ) = struct.unpack_from('>2H2f', data, pos); pos += struct.calcsize('>2H2f')

        self.fieldCollisionType = FieldCollisionReaction(fieldCollisionType)

        assert fieldCollisionIsWorld in (0, 1)
        self.fieldCollisionIsWorld = bool(fieldCollisionIsWorld)

        assert self.fieldCollisionCoef >= 0.0

    def save(self) -> bytes:
        return struct.pack(
            '>2H2f',
            self.fieldCollisionType,
            int(self.fieldCollisionIsWorld),
            self.fieldCollisionCoord,
            self.fieldCollisionCoef
        )

    def toDict(self, enable: bool) -> DictGeneric:
        return {
            "enable": {
                "description": "Enable collision surface?",
                "value": enable
            },
            "type": {
                "description": "Type of particles reaction to collision. Possible types are:\n" \
                               "Cessation:  Particle stops at collision surface and stays alive for only 1 more frame.\n" \
                               "Reflection: Particle velocity Y value reflects (i.e., it is multiplied by -1).",
                "value": self.fieldCollisionType.name
            },
            "isWorld": {
                "description": "True: Collision is processed in world-space (global coordinates).\n" \
                               "False: Collision is processed in local-space (relative to the emitter).",
                "value": self.fieldCollisionIsWorld
            },
            "surfPosY": {
                "description": "Y position of the collision surface (in space defined by `isWorld`).",
                "value": self.fieldCollisionCoord
            },
            "reflectCoef": {
                "description": "For the `Reflection` type, coefficient of reflection. After the particle velocity Y value is reflected, it is scaled by this value (i.e., this value sets the bounciness of particles). Value must be positive.",
                "value": self.fieldCollisionCoef
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        enable = VerifyBool(dic["enable"]["value"])
        self.fieldCollisionType = FieldCollisionReaction[dic["type"]["value"]]
        self.fieldCollisionIsWorld = VerifyBool(dic["isWorld"]["value"])
        self.fieldCollisionCoord = VerifyF32(dic["surfPosY"]["value"])
        self.fieldCollisionCoef = VerifyF32Positive(dic["reflectCoef"]["value"])

        return enable


DefaultFieldCollisionData = nw__eft__FieldCollisionData()
DefaultFieldCollisionData.fieldCollisionType = FieldCollisionReaction.Cessation
DefaultFieldCollisionData.fieldCollisionIsWorld = False
DefaultFieldCollisionData.fieldCollisionCoord = 0.0
DefaultFieldCollisionData.fieldCollisionCoef = 0.0


class nw__eft__FieldConvergenceData:
    structSize = VEC3_SIZE + F32_SIZE
    assert structSize == 0x10

    fieldConvergencePos: Vec3Numeric
    fieldConvergenceRatio: Numeric

    def load(self, data: ByteString, pos: int = 0) -> None:
        # Convergence position
        self.fieldConvergencePos = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Convergence rate
        self.fieldConvergenceRatio = struct.unpack_from(F32_FMT, data, pos)[0]
        assert self.fieldConvergenceRatio >= 0.0

    def save(self) -> bytes:
        return b''.join((
            struct.pack(VEC3_FMT, *self.fieldConvergencePos),
            struct.pack(F32_FMT, self.fieldConvergenceRatio)
        ))

    def toDict(self, enable: bool) -> DictGeneric:
        return {
            "enable": {
                "description": "Enable convergence field?\n" \
                               "Convergence field is similar to an attractive magnetic field. However, the magnetic field is more dynamic as it impacts the particle velocity, whereas the convergence field forces particles to move in a straightforward, linear motion towards the target (convergence position).",
                "value": enable
            },
            "rate": {
                "description": "Specifies the rate of convergence. Value must be positive.\n" \
                               "(This value is affected by the emitter `momentumRnd` attribute.)",
                "value": self.fieldConvergenceRatio
            },
            "pos": {
                "description": "The position of convergence, in the local space of the emitter.",
                "value": Vec3ToDict(self.fieldConvergencePos)
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        enable = VerifyBool(dic["enable"]["value"])
        self.fieldConvergenceRatio = VerifyF32Positive(dic["rate"]["value"])
        self.fieldConvergencePos = Vec3FromDict(dic["pos"]["value"])

        return enable


DefaultFieldConvergenceData = nw__eft__FieldConvergenceData()
DefaultFieldConvergenceData.fieldConvergencePos = (0.0, 0.0, 0.0)
DefaultFieldConvergenceData.fieldConvergenceRatio = 0.0


class nw__eft__FieldPosAddData:
    structSize = VEC3_SIZE
    assert structSize == 0xC

    fieldPosAdd: Vec3Numeric

    def load(self, data: ByteString, pos: int = 0) -> None:
        # Position addition value
        self.fieldPosAdd = struct.unpack_from(VEC3_FMT, data, pos)

    def save(self) -> bytes:
        return struct.pack(VEC3_FMT, *self.fieldPosAdd)

    def toDict(self, enable: bool) -> DictGeneric:
        return {
            "enable": {
                "description": "Enable constant-additional-velocity field?",
                "value": enable
            },
            "vel": {
                "description": "A constant velocity that is always added to the particle position when this field is enabled.",
                "value": Vec3ToDict(self.fieldPosAdd)
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        enable = VerifyBool(dic["enable"]["value"])
        self.fieldPosAdd = Vec3FromDict(dic["vel"]["value"])

        return enable


DefaultFieldPosAddData = nw__eft__FieldPosAddData()
DefaultFieldPosAddData.fieldPosAdd = (0.0, 0.0, 0.0)


class nw__eft__FieldData:
    randomData: Optional[nw__eft__FieldRandomData]
    magnetData: Optional[nw__eft__FieldMagnetData]
    spinData: Optional[nw__eft__FieldSpinData]
    collisionData: Optional[nw__eft__FieldCollisionData]
    convergenceData: Optional[nw__eft__FieldConvergenceData]
    posAddData: Optional[nw__eft__FieldPosAddData]

    def load(self, data: ByteString, fieldFlg: int, pos: int = 0) -> int:
        # Random field
        if fieldFlg & FieldMask.Random:
            self.randomData = nw__eft__FieldRandomData()
            self.randomData.load(data, pos); pos += nw__eft__FieldRandomData.structSize
        else:
            self.randomData = None

        # Magnetic field
        if fieldFlg & FieldMask.Magnet:  # Unused in NSMBU
            self.magnetData = nw__eft__FieldMagnetData()
            self.magnetData.load(data, pos); pos += nw__eft__FieldMagnetData.structSize
        else:
            self.magnetData = None

        # Spin field
        if fieldFlg & FieldMask.Spin:
            self.spinData = nw__eft__FieldSpinData()
            self.spinData.load(data, pos); pos += nw__eft__FieldSpinData.structSize
        else:
            self.spinData = None

        # Collision field
        if fieldFlg & FieldMask.Collision:  # Unused in NSMBU
            self.collisionData = nw__eft__FieldCollisionData()
            self.collisionData.load(data, pos); pos += nw__eft__FieldCollisionData.structSize
        else:
            self.collisionData = None

        # Convergence field
        if fieldFlg & FieldMask.Convergence:  # Unused in NSMBU
            self.convergenceData = nw__eft__FieldConvergenceData()
            self.convergenceData.load(data, pos); pos += nw__eft__FieldConvergenceData.structSize
        else:
            self.convergenceData = None

        # Position addition field
        if fieldFlg & FieldMask.PosAdd:  # Unused in NSMBU
            self.posAddData = nw__eft__FieldPosAddData()
            self.posAddData.load(data, pos); pos += nw__eft__FieldPosAddData.structSize
        else:
            self.posAddData = None

        return pos

    def getFieldFlg(self) -> int:
        return (
            (self.randomData        is not None) * FieldMask.Random         |
            (self.magnetData        is not None) * FieldMask.Magnet         |
            (self.spinData          is not None) * FieldMask.Spin           |
            (self.collisionData     is not None) * FieldMask.Collision      |
            (self.convergenceData   is not None) * FieldMask.Convergence    |
            (self.posAddData        is not None) * FieldMask.PosAdd
        )

    def save(self) -> bytes:
        ret = bytearray()

        if self.randomData is not None:
            ret += self.randomData.save()

        if self.magnetData is not None:
            ret += self.magnetData.save()

        if self.spinData is not None:
            ret += self.spinData.save()

        if self.collisionData is not None:
            ret += self.collisionData.save()

        if self.convergenceData is not None:
            ret += self.convergenceData.save()

        if self.posAddData is not None:
            ret += self.posAddData.save()

        return bytes(ret)


class nw__eft__FluctuationData:
    fluctuationScale: Numeric
    fluctuationFreq: Numeric
    fluctuationPhaseRnd: bool

    def load(self, data: ByteString, pos: int = 0) -> int:
        basePos = pos

        (
            self.fluctuationScale,  # Amplitude
            self.fluctuationFreq,   # Frequency
            fluctuationPhaseRnd     # Random phase?
        ) = struct.unpack_from('>2fI', data, pos); pos += struct.calcsize('>2fI')

        assert self.fluctuationScale >= 0.0
        assert self.fluctuationFreq > 0.0

        assert fluctuationPhaseRnd in (0, 1)
        self.fluctuationPhaseRnd = bool(fluctuationPhaseRnd)

        assert pos - basePos == 0xC
        return pos

    def save(self) -> bytes:
        return struct.pack(
            '>2fI',
            self.fluctuationScale,
            self.fluctuationFreq,
            int(self.fluctuationPhaseRnd)
        )

    def toDict(self, enable: bool, applyAlpha: bool, applyScale: bool) -> DictGeneric:
        return {
            "enable": {
                "description": "Enable fluctuation effect?",
                "value": enable
            },
            "applyAlpha": {
                "description": "Apply effect to particle alpha?",
                "value": applyAlpha
            },
            "applyScale": {
                "description": "Apply effect to particle scale?",
                "value": applyScale
            },
            "amplitude": {
                "description": "Amplitude of fluctuation sine wave. Value must be positive.",
                "value": self.fluctuationScale
            },
            "frequency": {
                "description": "Frequency of fluctuation sine wave (reciprocal of period in frames). Value must be greater than zero.",
                "value": self.fluctuationFreq
            },
            "startRandPhase": {
                "description": "Start fluctuation at a random phase?",
                "value": self.fluctuationPhaseRnd
            }
        }

    def fromDict(self, dic: DictGeneric) -> Tuple[bool, bool, bool]:
        enable = VerifyBool(dic["enable"]["value"])
        applyAlpha = VerifyBool(dic["applyAlpha"]["value"])
        applyScale = VerifyBool(dic["applyScale"]["value"])
        self.fluctuationScale = VerifyF32Positive(dic["amplitude"]["value"])
        self.fluctuationFreq = VerifyF32PositiveNonZero(dic["frequency"]["value"])
        self.fluctuationPhaseRnd = VerifyBool(dic["startRandPhase"]["value"])

        return enable, applyAlpha, applyScale


DefaultFluctuationData = nw__eft__FluctuationData()
DefaultFluctuationData.fluctuationScale = 1.0
DefaultFluctuationData.fluctuationFreq = 1 / 20
DefaultFluctuationData.fluctuationPhaseRnd = False


class nw__eft__StripeData:
    stripeType: StripeType
    stripeOption: StripeOption
    stripeConnectOpt: StripeConnectOption
    stripeTexCoordOpt: StripeTexCoordOption
    stripeNumHistory: int
    stripeDivideNum: int
    stripeStartAlpha: Numeric
    stripeEndAlpha: Numeric
    stripeHistoryStep: int
    stripeHistoryInterpolate: Numeric
    stripeDirInterpolate: Numeric

    def load(self, data: ByteString, pos: int = 0) -> int:
        basePos = pos

        (
            stripeType,             # Type
            stripeOption,           # Option
            stripeConnectOpt,       # Connection type
            stripeTexCoordOpt,      # Texturing Option
            self.stripeNumHistory,  # How many histories to take?
            self.stripeDivideNum,   # Division count
            self.stripeStartAlpha,  # Start alpha
            self.stripeEndAlpha     # End alpha
        ) = struct.unpack_from('>4I2i2f', data, pos); pos += struct.calcsize('>4I2i2f')

        assert stripeType != StripeType.Max
        self.stripeType = StripeType(stripeType)

        self.stripeOption = StripeOption(stripeOption)
        self.stripeConnectOpt = StripeConnectOption(stripeConnectOpt)
        self.stripeTexCoordOpt = StripeTexCoordOption(stripeTexCoordOpt)

        assert 2 <= self.stripeNumHistory <= 256
        assert 0 <= self.stripeDivideNum <= 10
        assert 0.0 <= self.stripeStartAlpha <= 1.0
        assert 0.0 <= self.stripeEndAlpha <= 1.0

        # UV scroll addition value (unused)
        stripeUVScroll = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE
        assert stripeUVScroll == (0.0, 0.0)

        (
            self.stripeHistoryStep,         # History tracing interval
            self.stripeHistoryInterpolate,  # History interpolation rate
            self.stripeDirInterpolate       # Direction interpolation rate
        ) = struct.unpack_from('>i2f', data, pos); pos += struct.calcsize('>i2f')

        assert 0 <= self.stripeHistoryStep <= 128
        assert 0.0 <= self.stripeHistoryInterpolate <= 1.0
        assert 0.0 <= self.stripeDirInterpolate <= 1.0

        assert pos - basePos == 0x34
        return pos

    def save(self) -> bytes:
        assert self.stripeType != StripeType.Max
        assert 2 <= self.stripeNumHistory <= 256
        assert 0 <= self.stripeDivideNum <= 10
        assert 0.0 <= self.stripeStartAlpha <= 1.0
        assert 0.0 <= self.stripeEndAlpha <= 1.0
        assert 0 <= self.stripeHistoryStep <= 128
        assert 0.0 <= self.stripeHistoryInterpolate <= 1.0
        assert 0.0 <= self.stripeDirInterpolate <= 1.0

        return b''.join((
            struct.pack(
                '>4I2i2f',
                self.stripeType,
                self.stripeOption,
                self.stripeConnectOpt,
                self.stripeTexCoordOpt,
                self.stripeNumHistory,
                self.stripeDivideNum,
                self.stripeStartAlpha,
                self.stripeEndAlpha
            ),
            struct.pack(VEC2_FMT, 0.0, 0.0),
            struct.pack(
                '>i2f',
                self.stripeHistoryStep,
                self.stripeHistoryInterpolate,
                self.stripeDirInterpolate
            )
        ))

    def toDict(self, stripeEmitterCoord: bool) -> DictGeneric:
        return {
            "calcType": {
                "description": "Stripe calculation method. Possible values are:\n" \
                               "Billboard:     This type of stripe always faces the camera.\n" \
                               "EmitterMatrix: This type of stripe faces the emitter.\n" \
                               "EmitterUpDown: This type of stripe stretches out vertically with respect to the emitter.",
                "value": self.stripeType.name
            },
            "followEmitter": {
                "description": "Should past samples follow the emitter when it is moved?",
                "value": stripeEmitterCoord
            },
            "shape": {
                "description": "Stripe shape. Possible values are:\n" \
                               "Normal: A flat polygon.\n" \
                               "Cross:  Two flat polygons crossed together. This option allows for the appearance of extra thickness when viewing the particle from multiple directions.",
                "value": self.stripeOption.name
            },
            "texCoordOpt": {
                "description": "Stripe texturing method. Possible values are:\n" \
                               "Full:     The texture image is applied matching the size of the stripe.\n" \
                               "Division: The texture image is applied to a rendering region of a fixed size.",
                "value": self.stripeTexCoordOpt.name
            },
            "divideNum": {
                "description": "Number of divisions. Sets the number of divisions for stripe particles. You can make stripe particles look smoother by increasing the number of divisions.\n" \
                               "Value must be in the range `[0, 10]`.",
                "value": self.stripeDivideNum
            },
            "startAlpha": {
                "description": "Starting point alpha. Sets the alpha value for the first vertex that is generated.\n" \
                               "Value must be in the range `[0, 1]`.",
                "value": self.stripeStartAlpha
            },
            "endAlpha": {
                "description": "Ending point alpha. Sets the alpha value for the last vertex that is generated.\n" \
                               "Value must be in the range `[0, 1]`.",
                "value": self.stripeEndAlpha
            },
            "dirInterpolate": {
                "description": "Directional interpolation ratio. Reduces distortions in stripes during sudden directional changes in the emitter by applying an offset to past samples. The smaller this value, the smaller the distortion of the stripes.\n" \
                               "Value must be in the range `[0, 1]`.",
                "value": self.stripeDirInterpolate
            },
            "historyParam": {
                "description": "Parameters for when the billboard type is set to `HistoricalStripe`.",
                "numHistory": {
                    "description": "Number of past samples to capture (i.e, length of stripe).\n" \
                                   "Positional information for particles after they have been emitted is recorded on a frame-by-frame basis for the specified number of frames. Stripes are generated from the recorded positional information, so the larger this value the longer the stripes.\n" \
                                   "Value must be in the range `[2, 256]`.",
                    "value": self.stripeNumHistory
                },
                "historyStep": {
                    "description": "Spacing at which to make past samples into polygons.\n" \
                                   "Specifies the number of frames to use as the interval for generating vertices from past samples. The smaller this value, the more smooth and fluid the stripe.\n" \
                                   "Value must be in the range `[0, 128]`.",
                    "value": self.stripeHistoryStep
                },
                "historyInterpolate": {
                    "description": "Slice interpolation ratio. Interpolates between the slices for a smooth connection. The smaller this value, the smoother a curve the particle makes in its rotational motion.\n" \
                                   "Only used when `followEmitter` is disabled.\n" \
                                   "Value must be in the range `[0, 1]`.",
                    "value": self.stripeHistoryInterpolate
                }
            },
            "consolidatedParam": {
                "description": "Parameters for when the billboard type is set to `ConsolidatedStripe`.",
                "connectType": {
                    "description": "Connection type. Possible types are:\n" \
                                   "Normal:  Standard stripe. At least two polygons are required to calculate the stripe.\n" \
                                   "Head:    Connects the starting point and end point of the stripe in a ring shape. At least three polygons are required to connect the stripe in a ring shape.\n" \
                                   "Emitter: Connects the end point of the stripe to the center of the emitter. At least two polygons are required to calculate the stripe.",
                    "value": self.stripeConnectOpt.name
                }
            }
        }

    def fromDict(self, dic: DictGeneric) -> bool:
        dic_historyParam: DictGeneric = dic["historyParam"]

        self.stripeType = StripeType[dic["calcType"]["value"]]; assert self.stripeType != StripeType.Max
        stripeEmitterCoord = VerifyBool(dic["followEmitter"]["value"])
        self.stripeOption = StripeOption[dic["shape"]["value"]]
        self.stripeTexCoordOpt = StripeTexCoordOption[dic["texCoordOpt"]["value"]]
        self.stripeDivideNum = VerifyIntRange(dic["divideNum"]["value"], 0, 10)
        self.stripeStartAlpha = VerifyF32Normal(dic["startAlpha"]["value"])
        self.stripeEndAlpha = VerifyF32Normal(dic["endAlpha"]["value"])
        self.stripeDirInterpolate = VerifyF32Normal(dic["dirInterpolate"]["value"])
        self.stripeNumHistory = VerifyIntRange(dic_historyParam["numHistory"]["value"], 2, 256)
        self.stripeHistoryStep = VerifyIntRange(dic_historyParam["historyStep"]["value"], 0, 128)
        self.stripeHistoryInterpolate = VerifyF32Normal(dic_historyParam["historyInterpolate"]["value"])
        self.stripeConnectOpt = StripeConnectOption[dic["consolidatedParam"]["connectType"]["value"]]

        return stripeEmitterCoord


DefaultStripeData = nw__eft__StripeData()
DefaultStripeData.stripeType = StripeType.Billboard
DefaultStripeData.stripeOption = StripeOption.Normal
DefaultStripeData.stripeConnectOpt = StripeConnectOption.Normal
DefaultStripeData.stripeTexCoordOpt = StripeTexCoordOption.Full
DefaultStripeData.stripeNumHistory = 60
DefaultStripeData.stripeDivideNum = 1
DefaultStripeData.stripeStartAlpha = 1.0
DefaultStripeData.stripeEndAlpha = 1.0
DefaultStripeData.stripeHistoryStep = 0
DefaultStripeData.stripeHistoryInterpolate = 1.0
DefaultStripeData.stripeDirInterpolate = 1.0


def TextureToDict(texRes: nw__eft__TextureRes, textureData: nw__eft__TextureEmitterData) -> DictGeneric:
    uvDict = UvShiftAnimToDict(textureData)

    return TextureToDictEx(
        texRes,
        textureData.numTexDivX, textureData.numTexDivY,
        textureData.texAddressingMode,
        uvDict,
        textureData.texPtnAnimMode,
        textureData.numTexPat,
        textureData.isTexPatAnimRand,
        textureData.texPatFreq,
        textureData.texPatTblUse,
        textureData.texPatTbl
    )


def TextureFromDict(texRes: nw__eft__TextureRes, textureData: nw__eft__TextureEmitterData, dic: DictGeneric) -> None:
    (
        textureData.numTexDivX, textureData.numTexDivY,
        textureData.texAddressingMode,
        uvDict,
        textureData.texPtnAnimMode,
        textureData.numTexPat,
        textureData.isTexPatAnimRand,
        textureData.texPatFreq,
        textureData.texPatTblUse,
        textureData.texPatTbl
    ) = TextureFromDictEx(texRes, dic)

    UvShiftAnimFromDict(textureData, uvDict)


class nw__eft__EmitterData:
    ### CommonEmitterData ###
    type: EmitterType
    randomSeed: int
    ptclDrawOrder: ParticleDrawOrder
    texture0ColorSource: ColorSource
    texture1ColorSource: ColorSource
    primitiveColorSource: ColorSource
    texture0AlphaSource: AlphaSource
    texture1AlphaSource: AlphaSource
    primitiveAlphaSource: AlphaSource
    userDataBit: int
    userDataU8: Tuple[int, int, int, int, int, int]
    userDataF: Tuple[Numeric, Numeric, Numeric, Numeric, Numeric, Numeric, Numeric, Numeric]
    userCallbackID: UserDataCallBackID
    namePos: int
    name: str
    texRes: Tuple[nw__eft__TextureRes, nw__eft__TextureRes]
    animKeyTable: nw__eft__AnimKeyTable
    primitiveFigure: nw__eft__PrimitiveFigure

    ### SimpleEmitterData ###
    isPolygon: bool
    isFollowAll: bool
    isEmitterBillboardMtx: bool
    isWorldGravity: bool
    isDirectional: bool
    isStopEmitInFade: bool
    volumeTblIndex: int
    volumeSweepStartRandom: bool
    isDisplayParent: bool
    emitDistEnabled: EmissionIntervalType
    isVolumeLatitudeEnabled: ArcOpeningType
    ptclRotType: PtclRotType
    ptclFollowType: PtclFollowType
    colorCombinerType: ColorCombinerType
    alphaBaseCombinerType: AlphaBaseCombinerType
    alphaCommonSource: AlphaCommonSource
    drawPath: int
    displaySide: DisplaySideType
    dynamicsRandom: Numeric
    transformSRT: Mtx34Numeric
    transformRT: Mtx34Numeric
    scale: Vec3Numeric
    rot: Vec3Numeric
    trans: Vec3Numeric
    rotRnd: Vec3Numeric
    transRnd: Vec3Numeric
    blendType: BlendType
    zBufATestType: ZBufATestType
    volumeType: VolumeType
    volumeRadius: Vec3Numeric
    volumeSweepStart: int
    volumeSweepParam: int
    volumeCaliber: Numeric
    volumeLatitude: Numeric
    volumeLatitudeDir: VolumeLatitudeDir
    lineCenter: Numeric
    formScale: Vec3Numeric
    color0: Color3Numeric
    color1: Color3Numeric
    alpha: Numeric
    emitDistUnit: Numeric
    emitDistMax: Numeric
    emitDistMin: Numeric
    emitDistMargin: Numeric
    emitRate: Numeric
    startFrame: int
    endFrame: int
    lifeStep: int
    lifeStepRnd: int
    figureVel: Numeric
    emitterVel: Numeric
    initVelRnd: Numeric
    emitterVelDir: Vec3Numeric
    emitterVelDirAngle: Numeric
    spreadVec: Vec3Numeric
    airRegist: Numeric
    gravity: Vec3Numeric
    xzDiffusionVel: Numeric
    initPosRand: Numeric
    ptclLife: int
    ptclLifeRnd: int
    meshType: MeshType
    billboardType: BillboardType
    rotBasis: Vec2Numeric
    toCameraOffset: Numeric
    textureData: Tuple[nw__eft__TextureEmitterData, nw__eft__TextureEmitterData]
    colorCalcType: Tuple[ColorCalcType, ColorCalcType]
    color: Tuple[Tuple[Color3Numeric, Color3Numeric, Color3Numeric], Tuple[Color3Numeric, Color3Numeric, Color3Numeric]]
    colorSection1: Tuple[int, int]
    colorSection2: Tuple[int, int]
    colorSection3: Tuple[int, int]
    colorNumRepeat: Tuple[int, int]
    colorRepeatStartRand: Tuple[bool, bool]
    colorScale: Numeric
    initAlpha: Numeric
    diffAlpha21: Numeric
    diffAlpha32: Numeric
    alphaSection1: int
    alphaSection2: int
    texture1ColorBlend: ColorAlphaBlendType
    primitiveColorBlend: ColorAlphaBlendType
    texture1AlphaBlend: ColorAlphaBlendType
    primitiveAlphaBlend: ColorAlphaBlendType
    scaleSection1: int
    scaleSection2: int
    scaleRand: Numeric
    baseScale: Vec2Numeric
    initScale: Vec2Numeric
    diffScale21: Vec2Numeric
    diffScale32: Vec2Numeric
    initRot: Vec3Numeric
    initRotRand: Vec3Numeric
    rotVel: Vec3Numeric
    rotVelRand: Vec3Numeric
    rotRegist: Numeric
    alphaAddInFade: Numeric
    shaderType: FragmentShaderVariation
    userShaderSetting: int
    shaderUseSoftEdge: bool
    shaderApplyAlphaToRefract: bool
    shaderParam0: Numeric
    shaderParam1: Numeric
    softFadeDistance: Numeric
    softVolumeParam: Numeric
    userShaderDefine1: str
    userShaderDefine2: str
    userShaderFlag: int
    userShaderSwitchFlag: int
    userShaderParam: nw__eft__UserShaderParam

    ### ComplexEmitterData ###
    # (These attributes are only available if `self.type == EmitterType.Complex`.)
    childData: Optional[nw__eft__ChildData]
    fieldData: Optional[nw__eft__FieldData]
    fluctuationApplyAlpha: bool
    fluctuationApplyScale: bool
    fluctuationData: Optional[nw__eft__FluctuationData]
    stripeEmitterCoord: bool
    stripeData: Optional[nw__eft__StripeData]

    def __init__(self) -> None:
        # CommonEmitterData
        self.texRes = tuple(nw__eft__TextureRes() for _ in range(TextureSlot.BinMax))
        self.animKeyTable = nw__eft__AnimKeyTable()
        self.primitiveFigure = nw__eft__PrimitiveFigure()
        # SimpleEmitterData
        self.textureData = tuple(nw__eft__TextureEmitterData() for _ in range(TextureSlot.BinMax))
        self.userShaderParam = nw__eft__UserShaderParam()

    def load(self, data: ByteString, pos: int = 0) -> None:
        basePos = pos

        ### CommonEmitterData ###

        (
            type_,              # Type
            flg,                # General flags
            self.randomSeed,    # RNG seed
            userData,           # User data uint 1
            userData2           # User data uint 2
        ) = struct.unpack_from('>5I', data, pos); pos += struct.calcsize('>5I')

        self.type = EmitterType(type_)

        assert (flg & 0b11111111111111100000000111111111) == 0  # Ununsed bits

        assert (not (flg & EmitterFlg.EnableSortParticle)) or (not (flg & EmitterFlg.ReverseOrderParticle))
        if flg & EmitterFlg.EnableSortParticle:
            self.ptclDrawOrder = ParticleDrawOrder.ZSort        # Particles are drawn in sorted order by Z value
        elif flg & EmitterFlg.ReverseOrderParticle:
            self.ptclDrawOrder = ParticleDrawOrder.Descending   # Particles are drawn in descending order
        else:
            self.ptclDrawOrder = ParticleDrawOrder.Ascending    # Particles are drawin in ascending order

        self.texture0ColorSource = ColorSource(int((flg & EmitterFlg.Texture0ColorOne) != 0))    # Replace Texture 0 color with 1.0?
        self.texture1ColorSource = ColorSource(int((flg & EmitterFlg.Texture1ColorOne) != 0))    # Replace Texture 1 color with 1.0?
        self.primitiveColorSource = ColorSource(int((flg & EmitterFlg.PrimitiveColorOne) != 0))  # Replace Primitive color with 1.0?
        self.texture0AlphaSource = AlphaSource(int((flg & EmitterFlg.Texture0AlphaOne) != 0))    # Replace Texture 0 alpha with 1.0?
        self.texture1AlphaSource = AlphaSource(int((flg & EmitterFlg.Texture1AlphaOne) != 0))    # Replace Texture 1 alpha with 1.0?
        self.primitiveAlphaSource = AlphaSource(int((flg & EmitterFlg.PrimitiveAlphaOne) != 0))  # Replace Primitive alpha with 1.0?

        # User data (1/3: 16-bit bitfield)
        self.userDataBit = userData >> 16

        # User data (2/3: 1-byte number array)
        self.userDataU8 = (
            userData & 0xFF,
            userData >> 8 & 0xFF,
            userData2 & 0xFF,
            userData2 >> 8 & 0xFF,
            userData2 >> 16 & 0xFF,
            userData2 >> 24 & 0xFF
        )

        # User data (3/3: float number array)
        userDataF_fmt = f'>{UserDataParamIdx.Max}f'
        self.userDataF = struct.unpack_from(userDataF_fmt, data, pos); pos += struct.calcsize(userDataF_fmt)

        (
            userCallbackID, # User data callback ID (0-7, or -1 for none)
            self.namePos,   # Name (offset from beginning of name table)
          # self.name       # Name (set at runtime)
        ) = struct.unpack_from('>2i4x', data, pos); pos += struct.calcsize('>2i4x')

        self.userCallbackID = UserDataCallBackID(userCallbackID)

        # Texture resources
        assert len(self.texRes) == TextureSlot.BinMax
        for texRes in self.texRes:
            texRes.load(data, pos); pos += nw__eft__TextureRes.structSize

        # Keyframe animation resource
        self.animKeyTable.load(data, pos); pos += nw__eft__AnimKeyTable.structSize

        # Primitive to use
        self.primitiveFigure.load(data, pos); pos += nw__eft__PrimitiveFigure.structSize

        assert pos - basePos == 0x280  # CommonEmitterData

        ### SimpleEmitterData ###

        (
            isPolygon,                  # Unused (Is polygon?)
            isFollowAll,                # Unused
            isEmitterBillboardMtx,      # Unused (Can the billboard matrix be set per emitter?)
            isWorldGravity,             # Emission: Apply gravity in world coordinates?
            isDirectional,              # Unused
            isStopEmitInFade,           # Fade out: Stop emitting during fade?
            self.volumeTblIndex,        # Emitter shape: Index when using volume table
            volumeSweepStartRandom,     # Emitter shape: Randomize arc start angle
            isDisplayParent,            # Draw the parent
            emitDistEnabled,            # Equidistant emission: enabled?
            isVolumeLatitudeEnabled,    # Sphere latitude: enabled?
            ptclRotType,                # Rotation type
            ptclFollowType,             # Follow type
            colorCombinerType,          # Color combiner type
            alphaCombinerType,          # Alpha combiner type
            self.drawPath,              # Draw path
            displaySide,                # Display side
            self.dynamicsRandom         # Dynamics random factor
        ) = struct.unpack_from('>11Bx4IiIf', data, pos); pos += struct.calcsize('>11Bx4IiIf')

        assert isPolygon in (0, 1)
        self.isPolygon = bool(isPolygon)

        assert isFollowAll in (0, 1)
        self.isFollowAll = bool(isFollowAll)

        assert isEmitterBillboardMtx in (0, 1)
        self.isEmitterBillboardMtx = bool(isEmitterBillboardMtx)

        assert isWorldGravity in (0, 1)
        self.isWorldGravity = bool(isWorldGravity)

        assert isDirectional in (0, 1)
        self.isDirectional = bool(isDirectional)

        assert isStopEmitInFade in (0, 1)
        self.isStopEmitInFade = bool(isStopEmitInFade)

        assert volumeSweepStartRandom in (0, 1)
        self.volumeSweepStartRandom = bool(volumeSweepStartRandom)

        assert isDisplayParent in (0, 1)
        self.isDisplayParent = bool(isDisplayParent)

        self.emitDistEnabled = EmissionIntervalType(emitDistEnabled)
        self.isVolumeLatitudeEnabled = ArcOpeningType(isVolumeLatitudeEnabled)
        self.ptclRotType = PtclRotType(ptclRotType)
        self.ptclFollowType = PtclFollowType(ptclFollowType)
        self.colorCombinerType = ColorCombinerType(colorCombinerType)
        self.alphaBaseCombinerType, self.alphaCommonSource = AlphaCombinerType(alphaCombinerType).deconstruct()

        assert 0 <= self.drawPath < 32

        self.displaySide = DisplaySideType(displaySide)

        assert self.dynamicsRandom >= 0.0

        # Emitter transformation matrix SRT
        self.transformSRT = tuple(struct.unpack_from(VEC4_FMT, data, pos + i * VEC4_SIZE) for i in range(3)); pos += 3 * VEC4_SIZE

        # Emitter transformation matrix RT
        self.transformRT = tuple(struct.unpack_from(VEC4_FMT, data, pos + i * VEC4_SIZE) for i in range(3)); pos += 3 * VEC4_SIZE

        # Emitter scale
        self.scale = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        assert self.scale[0] >= 0.0 and \
               self.scale[1] >= 0.0 and \
               self.scale[2] >= 0.0

        # Emitter rotation
        rotX: float
        rotY: float
        rotZ: float
        (
            rotX,
            rotY,
            rotZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        rotX = F32StandardToRegular(rotX, MATH_PI_2_STD, MATH_PI_2)
        assert 0.0 <= rotX <= MATH_PI_2

        rotY = F32StandardToRegular(rotY, MATH_PI_2_STD, MATH_PI_2)
        assert 0.0 <= rotY <= MATH_PI_2

        rotZ = F32StandardToRegular(rotZ, MATH_PI_2_STD, MATH_PI_2)
        assert 0.0 <= rotZ <= MATH_PI_2

        self.rot = (rotX, rotY, rotZ)

        # Emitter translation
        self.trans = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Emitter rotation random factor
        rotRndX: float
        rotRndY: float
        rotRndZ: float
        (
            rotRndX,
            rotRndY,
            rotRndZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        rotRndX = F32StandardToRegular(rotRndX, MATH_PI_STD, MATH_PI)
        assert 0.0 <= rotRndX <= MATH_PI

        rotRndY = F32StandardToRegular(rotRndY, MATH_PI_STD, MATH_PI)
        assert 0.0 <= rotRndY <= MATH_PI

        rotRndZ = F32StandardToRegular(rotRndZ, MATH_PI_STD, MATH_PI)
        assert 0.0 <= rotRndZ <= MATH_PI

        self.rotRnd = (rotRndX, rotRndY, rotRndZ)

        # Emitter translation random factor
        self.transRnd = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        assert self.transRnd[0] >= 0.0 and \
               self.transRnd[1] >= 0.0 and \
               self.transRnd[2] >= 0.0

        (
            blendType,      # Blend type
            zBufATestType,  # Z-buffer, alpha test type
            volumeType      # Volume type
        ) = struct.unpack_from('>3I', data, pos); pos += struct.calcsize('>3I')

        self.blendType = BlendType(blendType)
        self.zBufATestType = ZBufATestType(zBufATestType)

        self.volumeType = VolumeType(volumeType)
        if self.volumeType in (VolumeType.SphereSameDivide, VolumeType.SphereSameDivide64):
            assert self.isVolumeLatitudeEnabled == 0  # In this case, 0 is stored and the value is never used
            if self.volumeType == VolumeType.SphereSameDivide:
                assert 0 <= self.volumeTblIndex < 8
            else:
                assert 0 <= self.volumeTblIndex < 61

        # Volume radius
        self.volumeRadius = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        assert self.volumeRadius[0] >= 0.0 and \
               self.volumeRadius[1] >= 0.0 and \
               self.volumeRadius[2] >= 0.0

        (
            self.volumeSweepStart,  # Arc width (start)
            self.volumeSweepParam,  # Arc calculation parameter (varies based on volume type)
            self.volumeCaliber,     # Inner diameter (Circle (filled), Sphere (filled))
            volumeLatitude          # Sphere latitude (0 = Whole)
        ) = struct.unpack_from('>iI2f', data, pos); pos += struct.calcsize('>iI2f')

        assert 0.0 <= self.volumeCaliber <= 1.0

        self.volumeLatitude = F32StandardToRegular(volumeLatitude, MATH_PI_STD, MATH_PI)
        assert 0.0 <= self.volumeLatitude <= MATH_PI

        # Sphere latitude direction
        volumeLatitudeDir = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        self.volumeLatitudeDir = VolumeLatitudeDir(volumeLatitudeDir)

        # Center position for line volume type
        self.lineCenter = struct.unpack_from(F32_FMT, data, pos)[0]; pos += F32_SIZE

        # Emitter form scale
        self.formScale = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        assert self.formScale[0] >= 0.0 and \
               self.formScale[1] >= 0.0 and \
               self.formScale[2] >= 0.0

        # Emitter global multiply color 0
        color0 = struct.unpack_from(VEC4_FMT, data, pos); pos += VEC4_SIZE
        assert color0[3] == 1.0
        self.color0 = color0[:3]

        # Emitter global multiply color 1
        color1 = struct.unpack_from(VEC4_FMT, data, pos); pos += VEC4_SIZE
        assert color1[3] == 1.0
        self.color1 = color1[:3]

        (
            self.alpha,             # Emitter global multiply alpha
            self.emitDistUnit,      # Distance emission: Emission distance
            self.emitDistMax,       # Distance emission: Maximum addition value in 1 frame
            self.emitDistMin,       # Distance emission: Minimum addition value in 1 frame
            self.emitDistMargin,    # Distance emission: Travel distance cutoff threshold
            self.emitRate,          # Emission rate
            self.startFrame,        # Emission start frame
            self.endFrame,          # Emission end frame
            self.lifeStep,          # Emission interval
            self.lifeStepRnd,       # Emission interval random factor
            self.figureVel,         # Omnidirectional velocity
            self.emitterVel,        # Unidirectional velocity
            self.initVelRnd         # Initial velocity randomizer factor
        ) = struct.unpack_from('>6f4i3f', data, pos); pos += struct.calcsize('>6f4i3f')

        assert self.alpha >= 0.0
        assert self.emitDistUnit > 0.0
        assert 0.0 <= self.emitDistMin <= self.emitDistMax
        assert self.emitDistMargin >= 0.0
        assert self.emitRate >= 0.0
        assert self.startFrame >= 0
        assert self.endFrame >= self.startFrame
        assert self.lifeStep >= 0
        assert self.lifeStepRnd >= 0
        assert self.emitterVel >= 0.0
        assert self.initVelRnd >= 0.0

        # Direction of unidirectional velocity
        self.emitterVelDir = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        assert -1.0 <= self.emitterVelDir[0] <= 1.0 and \
               -1.0 <= self.emitterVelDir[1] <= 1.0 and \
               -1.0 <= self.emitterVelDir[2] <= 1.0

        # Diffusion angle in unidirectional velocity
        self.emitterVelDirAngle = struct.unpack_from(F32_FMT, data, pos)[0]; pos += F32_SIZE
        assert 0.0 <= self.emitterVelDirAngle <= 180.0

        # Spread/diffusion vector
        self.spreadVec = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE
        assert self.spreadVec[0] >= 0.0 and \
               self.spreadVec[1] >= 0.0 and \
               self.spreadVec[2] >= 0.0

        # Air resistance
        self.airRegist = struct.unpack_from(F32_FMT, data, pos)[0]; pos += F32_SIZE
        assert self.airRegist >= 0.0

        # Gravity
        self.gravity = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        (
            self.xzDiffusionVel,    # Y-axis diffusion velocity
            self.initPosRand,       # Initial position randomizer factor
            self.ptclLife,          # Particle life
            self.ptclLifeRnd,       # Particle life random factor
            meshType,               # Mesh type (particle or primitive)
            billboardType           # Billboard type
        ) = struct.unpack_from('>2f2i2I', data, pos); pos += struct.calcsize('>2f2i2I')

        assert self.initPosRand >= 0.0
        assert self.ptclLife > 0
        assert 0 <= self.ptclLifeRnd <= self.ptclLife

        self.meshType = MeshType(meshType)
        if self.meshType == MeshType.Primitive:
            assert self.primitiveFigure.index != 0xFFFFFFFF
        else:
            assert self.primitiveFigure.index == 0xFFFFFFFF
            if self.meshType == MeshType.Stripe:
                assert self.type == EmitterType.Complex
            else:
                assert self.meshType == MeshType.Particle

        assert billboardType != BillboardType.Primitive
        self.billboardType = BillboardType(billboardType)
        if (self.billboardType == BillboardType.HistoricalStripe or
            self.billboardType == BillboardType.ConsolidatedStripe):
            assert self.type == EmitterType.Complex

        # Rotation basis
        self.rotBasis = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # Drawing offset in camera direction
        self.toCameraOffset = struct.unpack_from(F32_FMT, data, pos)[0]; pos += F32_SIZE

        # Texture data
        assert len(self.textureData) == TextureSlot.BinMax
        for textureData in self.textureData:
            textureData.load(data, pos); pos += nw__eft__TextureEmitterData.structSize

        # Color calculation type (per kind)
        self.colorCalcType = tuple(ColorCalcType(struct.unpack_from(U32_FMT, data, pos + i * U32_SIZE)[0]) for i in range(ColorKind.Max)); pos += U32_SIZE * ColorKind.Max

        # Color (per kind): Up to 3 colors each
        color = tuple(tuple(struct.unpack_from(VEC4_FMT, data, pos + (i * 3 + j) * VEC4_SIZE) for j in range(3)) for i in range(ColorKind.Max)); pos += VEC4_SIZE * 3 * ColorKind.Max
        for array in color:
            for elem in array:
                assert elem[3] == 1.0
        self.color = tuple(tuple(elem[:3] for elem in array) for array in color)

        # Length (%) of first color section (per kind)
        self.colorSection1 = tuple(struct.unpack_from(S32_FMT, data, pos + i * S32_SIZE)[0] for i in range(ColorKind.Max)); pos += S32_SIZE * ColorKind.Max

        # Length (%) of middle color section (per kind)
        self.colorSection2 = tuple(struct.unpack_from(S32_FMT, data, pos + i * S32_SIZE)[0] for i in range(ColorKind.Max)); pos += S32_SIZE * ColorKind.Max

        # Length (%) of last color section (per kind)
        self.colorSection3 = tuple(struct.unpack_from(S32_FMT, data, pos + i * S32_SIZE)[0] for i in range(ColorKind.Max)); pos += S32_SIZE * ColorKind.Max

        # Color animation repetition count
        self.colorNumRepeat = tuple(struct.unpack_from(S32_FMT, data, pos + i * S32_SIZE)[0] for i in range(ColorKind.Max)); pos += S32_SIZE * ColorKind.Max

        for colorSection1 in self.colorSection1:
            assert 0 <= colorSection1 <= 100

        for colorSection2 in self.colorSection2:
            assert 0 <= colorSection2 <= 100

        for colorSection3 in self.colorSection3:
            assert 0 <= colorSection3 <= 100

        for colorNumRepeat in self.colorNumRepeat:
            assert colorNumRepeat > 0

        # Set color repeat start position randomly?
        colorRepeatStartRand = tuple(struct.unpack_from(S32_FMT, data, pos + i * S32_SIZE)[0] for i in range(ColorKind.Max)); pos += S32_SIZE * ColorKind.Max
        for rand in colorRepeatStartRand:
            assert rand in (0, 1)
        self.colorRepeatStartRand = tuple(bool(rand) for rand in colorRepeatStartRand)

        (
            self.colorScale,        # Color scale
            self.initAlpha,         # Particle alpha 3-value 4-key animation: Alpha initial value (alpha1)
            self.diffAlpha21,       # Particle alpha 3-value 4-key animation: alpha2 - alpha1
            self.diffAlpha32,       # Particle alpha 3-value 4-key animation: alpha3 - alpha2
            self.alphaSection1,     # Particle alpha 3-value 4-key animation: First section length (%)
            self.alphaSection2,     # Particle alpha 3-value 4-key animation: Second section length (%)
            texture1ColorBlend,     # Subtexture color composite type
            primitiveColorBlend,    # Primitive color composite type
            texture1AlphaBlend,     # Subtexture alpha composite type
            primitiveAlphaBlend,    # Primitive alpha composite type
            self.scaleSection1,     # Particle scale 3-value 4-key animation: First section length (%)
            self.scaleSection2,     # Particle scale 3-value 4-key animation: Second section length (%)
            self.scaleRand          # Particle scale: Scale randomizer factor
        ) = struct.unpack_from('>4f2i4I2if', data, pos); pos += struct.calcsize('>4f2i4I2if')

        assert self.colorScale >= 0.0
        assert 0.0 <= self.initAlpha <= 1.0

        alpha2: float = self.initAlpha + self.diffAlpha21
        alpha3: float = alpha2 + self.diffAlpha32
        assert 0.0 <= alpha2 <= 1.0
        assert 0.0 <= alpha3 <= 1.0

        assert 0 <= self.alphaSection1 <= 100
        assert 0 <= self.alphaSection2 <= 100
        assert 0 <= self.scaleSection1 <= 100 or self.scaleSection1 == -127
        assert 0 <= self.scaleSection2 <= 100

        self.texture1ColorBlend = ColorAlphaBlendType(texture1ColorBlend)
        self.primitiveColorBlend = ColorAlphaBlendType(primitiveColorBlend)
        self.texture1AlphaBlend = ColorAlphaBlendType(texture1AlphaBlend)
        self.primitiveAlphaBlend = ColorAlphaBlendType(primitiveAlphaBlend)

        # Particle Scale: Scale base value
        self.baseScale = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # Particle scale 3-value 4-key animation: Scale initial value (scale1) (%)
        self.initScale = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # Particle scale 3-value 4-key animation: scale2 - scale1 (%)
        self.diffScale21 = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # Particle scale 3-value 4-key animation: scale3 - scale2 (%)
        self.diffScale32 = struct.unpack_from(VEC2_FMT, data, pos); pos += VEC2_SIZE

        # Particle rotation: Initial rotation
        initRotX: float
        initRotY: float
        initRotZ: float
        (
            initRotX,
            initRotY,
            initRotZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Particle rotation: Initial rotation random factor
        initRotRandX: float
        initRotRandY: float
        initRotRandZ: float
        (
            initRotRandX,
            initRotRandY,
            initRotRandZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Particle rotation: Rotation velocity
        rotVelX: float
        rotVelY: float
        rotVelZ: float
        (
            rotVelX,
            rotVelY,
            rotVelZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        # Particle rotation: Rotation velocity random factor
        rotVelRandX: float
        rotVelRandY: float
        rotVelRandZ: float
        (
            rotVelRandX,
            rotVelRandY,
            rotVelRandZ
        ) = struct.unpack_from(VEC3_FMT, data, pos); pos += VEC3_SIZE

        (
            self.rotRegist,             # Particle rotation: Rotation velocity decay factor
            self.alphaAddInFade,        # Fade out: Alpha addition value during fade
            shaderType,                 # Shader type
            self.userShaderSetting,     # User shader type
            shaderUseSoftEdge,          # Use soft edge?
            shaderApplyAlphaToRefract,  # Apply alpha to refraction shader?
            self.shaderParam0,          # Shader parameter 0
            self.shaderParam1,          # Shader parameter 1
            self.softFadeDistance,      # Soft edge: Fade distance
            self.softVolumeParam,       # Soft edge: Volume value
            userShaderDefine1,          # User shader compiler define 1
            userShaderDefine2,          # User shader compiler define 2
            self.userShaderFlag,        # User shader flags
            self.userShaderSwitchFlag   # User shader switch flags
        ) = struct.unpack_from('>2f4B4f16s16s2I', data, pos); pos += struct.calcsize('>2f4B4f16s16s2I')

        assert self.rotRegist >= 0.0
        assert self.alphaAddInFade >= 0.0

        self.shaderType = FragmentShaderVariation(shaderType)

        assert 0 <= self.userShaderSetting < 8

        assert shaderUseSoftEdge in (0, 1)
        self.shaderUseSoftEdge = bool(shaderUseSoftEdge)

        assert shaderApplyAlphaToRefract in (0, 1)
        self.shaderApplyAlphaToRefract = bool(shaderApplyAlphaToRefract)

        assert self.shaderParam0 >= 0.0
        assert self.shaderParam1 >= 0.0
        assert self.softFadeDistance >= 0.0
        assert self.softVolumeParam >= 0.0

        if self.ptclRotType == PtclRotType.NoWork:
            assert initRotX     == 0.0 and initRotY     == 0.0 and initRotZ     == 0.0
            assert initRotRandX == 0.0 and initRotRandY == 0.0 and initRotRandZ == 0.0
            assert rotVelX      == 0.0 and rotVelY      == 0.0 and rotVelZ      == 0.0
            assert rotVelRandX  == 0.0 and rotVelRandY  == 0.0 and rotVelRandZ  == 0.0

        elif self.ptclRotType == PtclRotType.RotX:
            assert initRotY     == 0.0 and initRotZ     == 0.0
            assert initRotRandY == 0.0 and initRotRandZ == 0.0
            assert rotVelY      == 0.0 and rotVelZ      == 0.0
            assert rotVelRandY  == 0.0 and rotVelRandZ  == 0.0

        elif self.ptclRotType == PtclRotType.RotY:
            assert initRotX     == 0.0 and initRotZ     == 0.0
            assert initRotRandX == 0.0 and initRotRandZ == 0.0
            assert rotVelX      == 0.0 and rotVelZ      == 0.0
            assert rotVelRandX  == 0.0 and rotVelRandZ  == 0.0

        elif self.ptclRotType == PtclRotType.RotZ:
            assert initRotX     == 0.0 and initRotY     == 0.0
            assert initRotRandX == 0.0 and initRotRandY == 0.0
            assert rotVelX      == 0.0 and rotVelY      == 0.0
            assert rotVelRandX  == 0.0 and rotVelRandY  == 0.0

        if self.ptclRotType in (PtclRotType.RotX, PtclRotType.RotXYZ):
            initRotX = F32StandardToRegular(initRotX, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= initRotX <= MATH_PI_2

            initRotRandX = F32StandardToRegular(initRotRandX, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= initRotRandX <= MATH_PI_2

            rotVelX = F32StandardToRegularMulti(rotVelX, MATH_PI_STD, MATH_PI, -MATH_PI_STD, -MATH_PI)
            assert -MATH_PI <= rotVelX <= MATH_PI

            rotVelRandX = F32StandardToRegular(rotVelRandX, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= rotVelRandX <= MATH_PI_2

        if self.ptclRotType in (PtclRotType.RotY, PtclRotType.RotXYZ):
            initRotY = F32StandardToRegular(initRotY, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= initRotY <= MATH_PI_2

            initRotRandY = F32StandardToRegular(initRotRandY, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= initRotRandY <= MATH_PI_2

            rotVelY = F32StandardToRegularMulti(rotVelY, MATH_PI_STD, MATH_PI, -MATH_PI_STD, -MATH_PI)
            assert -MATH_PI <= rotVelY <= MATH_PI

            rotVelRandY = F32StandardToRegular(rotVelRandY, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= rotVelRandY <= MATH_PI_2

        if self.ptclRotType in (PtclRotType.RotZ, PtclRotType.RotXYZ):
            initRotZ = F32StandardToRegular(initRotZ, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= initRotZ <= MATH_PI_2

            initRotRandZ = F32StandardToRegular(initRotRandZ, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= initRotRandZ <= MATH_PI_2

            rotVelZ = F32StandardToRegularMulti(rotVelZ, MATH_PI_STD, MATH_PI, -MATH_PI_STD, -MATH_PI)
            assert -MATH_PI <= rotVelZ <= MATH_PI

            rotVelRandZ = F32StandardToRegular(rotVelRandZ, MATH_PI_2_STD, MATH_PI_2)
            assert 0.0 <= rotVelRandZ <= MATH_PI_2

        self.initRot     = (initRotX,     initRotY,     initRotZ)
        self.initRotRand = (initRotRandX, initRotRandY, initRotRandZ)
        self.rotVel      = (rotVelX,      rotVelY,      rotVelZ)
        self.rotVelRand  = (rotVelRandX,  rotVelRandY,  rotVelRandZ)

        self.userShaderDefine1 = readString(userShaderDefine1)
        self.userShaderDefine2 = readString(userShaderDefine2)

        # User shader parameters
        self.userShaderParam.load(data, pos); pos += nw__eft__UserShaderParam.structSize

        assert pos - basePos == 0x6F4  # SimpleEmitterData

        ### ComplexEmitterData ###

        if self.type != EmitterType.Complex:
            attributes = (
                'childData',
                'fieldData',
                'fluctuationApplyAlpha', 'fluctuationApplyScale', 'fluctuationData',
                'stripeEmitterCoord', 'stripeData'
            )
            for attr in attributes:
                if hasattr(self, attr):
                    delattr(self, attr)
            return

        (
            childFlg,
            fieldFlg,
            fluctuationFlg,
            stripeFlg,
            childDataOffset,
            fieldDataOffset,
            fluctuationDataOffset,
            stripeDataOffset,
            emitterDataSize
        ) = struct.unpack_from('>I3H2x4Hi', data, pos); pos += struct.calcsize('>I3H2x4Hi')

        assert pos - basePos == 0x70C  # ComplexEmitterData

        maxPos = pos

        if childFlg & ChildFlg.Enable:
            assert childDataOffset == 0x70C
            self.childData = nw__eft__ChildData()
            maxPos = max(maxPos, self.childData.load(data, childFlg, pos))
        else:
            assert childFlg == 0x00000054
            assert childDataOffset == 0
            self.childData = None

        if fieldFlg != 0:
            assert fieldDataOffset >= 0x70C
            self.fieldData = nw__eft__FieldData()
            maxPos = max(maxPos, self.fieldData.load(data, fieldFlg, basePos + fieldDataOffset))
        else:
            assert fieldDataOffset == 0
            self.fieldData = None

        assert (fluctuationFlg & 0b11111111111111111111111111111000) == 0  # Ununsed bits
        self.fluctuationApplyAlpha = (fluctuationFlg & FluctuationFlg.ApplyAlpha) != 0
        self.fluctuationApplyScale = (fluctuationFlg & FluctuationFlg.ApplyScale) != 0

        if fluctuationFlg & FluctuationFlg.Enable:
            assert fluctuationDataOffset >= 0x70C
            self.fluctuationData = nw__eft__FluctuationData()
            maxPos = max(maxPos, self.fluctuationData.load(data, basePos + fluctuationDataOffset))
        else:
            assert fluctuationDataOffset == 0
            self.fluctuationData = None

        assert (stripeFlg & 0b11111111111111111111111111111110) == 0  # Ununsed bits
        self.stripeEmitterCoord = (stripeFlg & StripeFlg.EmitterCoord) != 0

        if (self.billboardType == BillboardType.HistoricalStripe or
            self.billboardType == BillboardType.ConsolidatedStripe):
            assert stripeDataOffset >= 0x70C
            self.stripeData = nw__eft__StripeData()
            maxPos = max(maxPos, self.stripeData.load(data, basePos + stripeDataOffset))
        else:
            assert stripeDataOffset == 0
            self.stripeData = None

        assert maxPos - basePos == emitterDataSize

    def save(self) -> bytes:
        userShaderDefine1 = self.userShaderDefine1.encode('shift_jis').ljust(16, b'\0')
        assert len(userShaderDefine1) == 16 and userShaderDefine1[-1] == 0

        userShaderDefine2 = self.userShaderDefine2.encode('shift_jis').ljust(16, b'\0')
        assert len(userShaderDefine2) == 16 and userShaderDefine2[-1] == 0

        assert len(self.texRes) == TextureSlot.BinMax
        assert len(self.textureData) == TextureSlot.BinMax

        flg = 0
        if self.ptclDrawOrder == ParticleDrawOrder.ZSort:
            flg |= EmitterFlg.EnableSortParticle
        elif self.ptclDrawOrder == ParticleDrawOrder.Descending:
            flg |= EmitterFlg.ReverseOrderParticle
        if self.texture0ColorSource == ColorSource.One:
            flg |= EmitterFlg.Texture0ColorOne
        if self.texture1ColorSource == ColorSource.One:
            flg |= EmitterFlg.Texture1ColorOne
        if self.primitiveColorSource == ColorSource.One:
            flg |= EmitterFlg.PrimitiveColorOne
        if self.texture0AlphaSource == AlphaSource.One:
            flg |= EmitterFlg.Texture0AlphaOne
        if self.texture1AlphaSource == AlphaSource.One:
            flg |= EmitterFlg.Texture1AlphaOne
        if self.primitiveAlphaSource == AlphaSource.One:
            flg |= EmitterFlg.PrimitiveAlphaOne

        assert 0 <= self.userDataBit <= 0xFFFF
        assert 0 <= self.userDataU8[0] <= 0xFF
        assert 0 <= self.userDataU8[1] <= 0xFF
        assert 0 <= self.userDataU8[2] <= 0xFF
        assert 0 <= self.userDataU8[3] <= 0xFF
        assert 0 <= self.userDataU8[4] <= 0xFF
        assert 0 <= self.userDataU8[5] <= 0xFF
        userData  = (self.userDataBit << 16 |
                     self.userDataU8[0]       | self.userDataU8[1] << 8)
        userData2 = (self.userDataU8[2]       | self.userDataU8[3] << 8 |
                     self.userDataU8[4] << 16 | self.userDataU8[5] << 24)

        if self.meshType == MeshType.Stripe:
            assert self.type == EmitterType.Complex

        assert 0 <= self.drawPath < 32

        assert 0.0 <= self.volumeLatitude <= MATH_PI

        (rotX, rotY, rotZ) = self.rot
        assert 0.0 <= rotX <= MATH_PI_2
        assert 0.0 <= rotY <= MATH_PI_2
        assert 0.0 <= rotZ <= MATH_PI_2

        (rotRndX, rotRndY, rotRndZ) = self.rotRnd
        assert 0.0 <= rotRndX <= MATH_PI
        assert 0.0 <= rotRndY <= MATH_PI
        assert 0.0 <= rotRndZ <= MATH_PI

        if self.ptclRotType == PtclRotType.NoWork:
            initRotX     = 0.0
            initRotY     = 0.0
            initRotZ     = 0.0
            initRotRandX = 0.0
            initRotRandY = 0.0
            initRotRandZ = 0.0
            rotVelX      = 0.0
            rotVelY      = 0.0
            rotVelZ      = 0.0
            rotVelRandX  = 0.0
            rotVelRandY  = 0.0
            rotVelRandZ  = 0.0

        elif self.ptclRotType == PtclRotType.RotX:
            initRotX     = self.initRot[0]
            initRotY     = 0.0
            initRotZ     = 0.0
            initRotRandX = self.initRotRand[0]
            initRotRandY = 0.0
            initRotRandZ = 0.0
            rotVelX      = self.rotVel[0]
            rotVelY      = 0.0
            rotVelZ      = 0.0
            rotVelRandX  = self.rotVelRand[0]
            rotVelRandY  = 0.0
            rotVelRandZ  = 0.0

        elif self.ptclRotType == PtclRotType.RotY:
            initRotX     = 0.0
            initRotY     = self.initRot[1]
            initRotZ     = 0.0
            initRotRandX = 0.0
            initRotRandY = self.initRotRand[1]
            initRotRandZ = 0.0
            rotVelX      = 0.0
            rotVelY      = self.rotVel[1]
            rotVelZ      = 0.0
            rotVelRandX  = 0.0
            rotVelRandY  = self.rotVelRand[1]
            rotVelRandZ  = 0.0

        elif self.ptclRotType == PtclRotType.RotZ:
            initRotX     = 0.0
            initRotY     = 0.0
            initRotZ     = self.initRot[2]
            initRotRandX = 0.0
            initRotRandY = 0.0
            initRotRandZ = self.initRotRand[2]
            rotVelX      = 0.0
            rotVelY      = 0.0
            rotVelZ      = self.rotVel[2]
            rotVelRandX  = 0.0
            rotVelRandY  = 0.0
            rotVelRandZ  = self.rotVelRand[2]

        else:
            (initRotX,     initRotY,     initRotZ)     = self.initRot
            (initRotRandX, initRotRandY, initRotRandZ) = self.initRotRand
            (rotVelX,      rotVelY,      rotVelZ)      = self.rotVel
            (rotVelRandX,  rotVelRandY,  rotVelRandZ)  = self.rotVelRand

        if self.ptclRotType in (PtclRotType.RotX, PtclRotType.RotXYZ):
            assert 0.0 <= initRotX <= MATH_PI_2
            assert 0.0 <= initRotRandX <= MATH_PI_2
            assert -MATH_PI <= rotVelX <= MATH_PI
            assert 0.0 <= rotVelRandX <= MATH_PI_2

        if self.ptclRotType in (PtclRotType.RotY, PtclRotType.RotXYZ):
            assert 0.0 <= initRotY <= MATH_PI_2
            assert 0.0 <= initRotRandY <= MATH_PI_2
            assert -MATH_PI <= rotVelY <= MATH_PI
            assert 0.0 <= rotVelRandY <= MATH_PI_2

        if self.ptclRotType in (PtclRotType.RotZ, PtclRotType.RotXYZ):
            assert 0.0 <= initRotZ <= MATH_PI_2
            assert 0.0 <= initRotRandZ <= MATH_PI_2
            assert -MATH_PI <= rotVelZ <= MATH_PI
            assert 0.0 <= rotVelRandZ <= MATH_PI_2

        ret = bytearray().join((
            ### CommonEmitterData ###
            struct.pack(
                '>5I',
                self.type,
                flg,
                self.randomSeed,
                userData,
                userData2
            ),
            struct.pack(f'>{UserDataParamIdx.Max}f', *self.userDataF),
            struct.pack(
                '>2i4x',
                self.userCallbackID,
                self.namePos,
              # self.name
            ),
            *(texRes.save() for texRes in self.texRes),
            self.animKeyTable.save(),
            self.primitiveFigure.save(),
            ### SimpleEmitterData ###
            struct.pack(
                '>11Bx4IiIf',
                int(self.isPolygon),
                int(self.isFollowAll),
                int(self.isEmitterBillboardMtx),
                int(self.isWorldGravity),
                int(self.isDirectional),
                int(self.isStopEmitInFade),
                self.volumeTblIndex,
                int(self.volumeSweepStartRandom),
                int(self.isDisplayParent),
                self.emitDistEnabled,
                self.isVolumeLatitudeEnabled,
                self.ptclRotType,
                self.ptclFollowType,
                self.colorCombinerType,
                AlphaCombinerType.construct(self.alphaBaseCombinerType, self.alphaCommonSource),
                self.drawPath,
                self.displaySide,
                self.dynamicsRandom
            ),
            *(struct.pack(VEC4_FMT, *(self.transformSRT[i])) for i in range(3)),
            *(struct.pack(VEC4_FMT, *(self.transformRT[i])) for i in range(3)),
            struct.pack(VEC3_FMT, *self.scale),
            struct.pack(VEC3_FMT, rotX, rotY, rotZ),
            struct.pack(VEC3_FMT, *self.trans),
            struct.pack(VEC3_FMT, rotRndX, rotRndY, rotRndZ),
            struct.pack(VEC3_FMT, *self.transRnd),
            struct.pack(
                '>3I',
                self.blendType,
                self.zBufATestType,
                self.volumeType
            ),
            struct.pack(VEC3_FMT, *self.volumeRadius),
            struct.pack(
                '>iI2f',
                self.volumeSweepStart,
                self.volumeSweepParam,
                self.volumeCaliber,
                self.volumeLatitude
            ),
            struct.pack(VEC3_FMT, *self.volumeLatitudeDir.value),
            struct.pack(F32_FMT, self.lineCenter),
            struct.pack(VEC3_FMT, *self.formScale),
            struct.pack(VEC4_FMT, *self.color0, 1.0),
            struct.pack(VEC4_FMT, *self.color1, 1.0),
            struct.pack(
                '>6f4i3f',
                self.alpha,
                self.emitDistUnit,
                self.emitDistMax,
                self.emitDistMin,
                self.emitDistMargin,
                self.emitRate,
                self.startFrame,
                self.endFrame,
                self.lifeStep,
                self.lifeStepRnd,
                self.figureVel,
                self.emitterVel,
                self.initVelRnd
            ),
            struct.pack(VEC3_FMT, *self.emitterVelDir),
            struct.pack(F32_FMT, self.emitterVelDirAngle),
            struct.pack(VEC3_FMT, *self.spreadVec),
            struct.pack(F32_FMT, self.airRegist),
            struct.pack(VEC3_FMT, *self.gravity),
            struct.pack(
                '>2f2i2I',
                self.xzDiffusionVel,
                self.initPosRand,
                self.ptclLife,
                self.ptclLifeRnd,
                self.meshType,
                self.billboardType
            ),
            struct.pack(VEC2_FMT, *self.rotBasis),
            struct.pack(F32_FMT, self.toCameraOffset),
            *(textureData.save() for textureData in self.textureData),
            *(struct.pack(U32_FMT, self.colorCalcType[i]) for i in range(ColorKind.Max)),
            *chain.from_iterable((struct.pack(VEC4_FMT, *(self.color[i][j]), 1.0) for j in range(3)) for i in range(ColorKind.Max)),
            *(struct.pack(S32_FMT, self.colorSection1[i]) for i in range(ColorKind.Max)),
            *(struct.pack(S32_FMT, self.colorSection2[i]) for i in range(ColorKind.Max)),
            *(struct.pack(S32_FMT, self.colorSection3[i]) for i in range(ColorKind.Max)),
            *(struct.pack(S32_FMT, self.colorNumRepeat[i]) for i in range(ColorKind.Max)),
            *(struct.pack(S32_FMT, int(self.colorRepeatStartRand[i])) for i in range(ColorKind.Max)),
            struct.pack(
                '>4f2i4I2if',
                self.colorScale,
                self.initAlpha,
                self.diffAlpha21,
                self.diffAlpha32,
                self.alphaSection1,
                self.alphaSection2,
                self.texture1ColorBlend,
                self.primitiveColorBlend,
                self.texture1AlphaBlend,
                self.primitiveAlphaBlend,
                self.scaleSection1,
                self.scaleSection2,
                self.scaleRand
            ),
            struct.pack(VEC2_FMT, *self.baseScale),
            struct.pack(VEC2_FMT, *self.initScale),
            struct.pack(VEC2_FMT, *self.diffScale21),
            struct.pack(VEC2_FMT, *self.diffScale32),
            struct.pack(VEC3_FMT, initRotX,     initRotY,     initRotZ),
            struct.pack(VEC3_FMT, initRotRandX, initRotRandY, initRotRandZ),
            struct.pack(VEC3_FMT, rotVelX,      rotVelY,      rotVelZ),
            struct.pack(VEC3_FMT, rotVelRandX,  rotVelRandY,  rotVelRandZ),
            struct.pack(
                '>2f4B4f16s16s2I',
                self.rotRegist,
                self.alphaAddInFade,
                self.shaderType,
                self.userShaderSetting,
                int(self.shaderUseSoftEdge),
                int(self.shaderApplyAlphaToRefract),
                self.shaderParam0,
                self.shaderParam1,
                self.softFadeDistance,
                self.softVolumeParam,
                userShaderDefine1,
                userShaderDefine2,
                self.userShaderFlag,
                self.userShaderSwitchFlag
            ),
            self.userShaderParam.save()
        ))

        assert len(ret) == 0x6F4  # SimpleEmitterData

        if self.type == EmitterType.Complex:
            ### ComplexEmitterData ###

            assert (((self.billboardType == BillboardType.HistoricalStripe or
                      self.billboardType == BillboardType.ConsolidatedStripe) and self.stripeData is not None) or
                    ((self.billboardType != BillboardType.HistoricalStripe and
                      self.billboardType != BillboardType.ConsolidatedStripe) and self.stripeData is None))

            childDataOffset = 0
            fieldDataOffset = 0
            fluctuationDataOffset = 0
            stripeDataOffset = 0

            childData = None
            fieldData = None
            fluctuationData = None
            stripeData = None

            childFlg = 0x00000054
            if self.childData is not None:
                childFlg = self.childData.getChildFlg()

            fieldFlg = 0
            if self.fieldData is not None:
                fieldFlg = self.fieldData.getFieldFlg()

            fluctuationFlg = (
                self.fluctuationApplyAlpha * FluctuationFlg.ApplyAlpha |
                self.fluctuationApplyScale * FluctuationFlg.ApplyScale
            )
            if self.fluctuationData is not None:
                fluctuationFlg |= FluctuationFlg.Enable

            stripeFlg = self.stripeEmitterCoord * StripeFlg.EmitterCoord

            emitterDataSize = 0x70C

            if self.childData is not None:
                childDataOffset = emitterDataSize
                childData = self.childData.save()
                emitterDataSize += len(childData)

            if fieldFlg != 0:
                fieldDataOffset = emitterDataSize
                fieldData = self.fieldData.save()
                emitterDataSize += len(fieldData)

            if self.fluctuationData is not None:
                fluctuationDataOffset = emitterDataSize
                fluctuationData = self.fluctuationData.save()
                emitterDataSize += len(fluctuationData)

            if self.stripeData is not None:
                stripeDataOffset = emitterDataSize
                stripeData = self.stripeData.save()
                emitterDataSize += len(stripeData)

            ret += struct.pack(
                '>I3H2x4Hi',
                childFlg,
                fieldFlg,
                fluctuationFlg,
                stripeFlg,
                childDataOffset,
                fieldDataOffset,
                fluctuationDataOffset,
                stripeDataOffset,
                emitterDataSize
            )

            assert len(ret) == 0x70C  # ComplexEmitterData

            if childDataOffset:
                ret += childData

            if fieldDataOffset:
                ret += fieldData

            if fluctuationDataOffset:
                ret += fluctuationData

            if stripeDataOffset:
                ret += stripeData

            assert len(ret) == emitterDataSize

        return bytes(ret)

    def toDict(self) -> DictGeneric:
        if self.type == EmitterType.Complex:
            assert (((self.billboardType == BillboardType.HistoricalStripe or
                      self.billboardType == BillboardType.ConsolidatedStripe) and self.stripeData is not None) or
                    ((self.billboardType != BillboardType.HistoricalStripe and
                      self.billboardType != BillboardType.ConsolidatedStripe) and self.stripeData is None))
        else:
            assert (self.billboardType != BillboardType.HistoricalStripe and
                    self.billboardType != BillboardType.ConsolidatedStripe)

        childEnable = False
        childData = DefaultChildData
        fieldRandomEnable = False
        fieldRandomData = DefaultFieldRandomData
        fieldMagnetEnable = False
        fieldMagnetData = DefaultFieldMagnetData
        fieldSpinEnable = False
        fieldSpinData = DefaultFieldSpinData
        fieldCollisionEnable = False
        fieldCollisionData = DefaultFieldCollisionData
        fieldConvergenceEnable = False
        fieldConvergenceData = DefaultFieldConvergenceData
        fieldPosAddEnable = False
        fieldPosAddData = DefaultFieldPosAddData
        fluctuationEnable = False
        fluctuationApplyAlpha = True
        fluctuationApplyScale = True
        fluctuationData = DefaultFluctuationData
        stripeEmitterCoord = False
        stripeData = DefaultStripeData
        if self.type == EmitterType.Complex:
            if self.childData is not None:
                childEnable = True
                childData = self.childData
            if self.fieldData is not None:
                if self.fieldData.randomData is not None:
                    fieldRandomEnable = True
                    fieldRandomData = self.fieldData.randomData
                if self.fieldData.magnetData is not None:
                    fieldMagnetEnable = True
                    fieldMagnetData = self.fieldData.magnetData
                if self.fieldData.spinData is not None:
                    fieldSpinEnable = True
                    fieldSpinData = self.fieldData.spinData
                if self.fieldData.collisionData is not None:
                    fieldCollisionEnable = True
                    fieldCollisionData = self.fieldData.collisionData
                if self.fieldData.convergenceData is not None:
                    fieldConvergenceEnable = True
                    fieldConvergenceData = self.fieldData.convergenceData
                if self.fieldData.posAddData is not None:
                    fieldPosAddEnable = True
                    fieldPosAddData = self.fieldData.posAddData
            fluctuationApplyAlpha = self.fluctuationApplyAlpha
            fluctuationApplyScale = self.fluctuationApplyScale
            if self.fluctuationData is not None:
                fluctuationEnable = True
                fluctuationData = self.fluctuationData
            stripeEmitterCoord = self.stripeEmitterCoord
            if self.stripeData is not None:
                stripeData = self.stripeData

        return {
            "general": {
                "emitterType": {
                    "description": "The type of the emitter. Possible types are:\n" \
                                   "Simple:  Allows use of the most basic features for creating an effect\n" \
                                   "Complex: Allows use of all fields, fluxes, child, and stripe features. Enabling these features will increase the processing cost in comparison with the simple type, but allows a wider range of effects to be created.",
                    "value": self.type.name
                },
                "ptclFollowType": {
                    "description": "Particle emitter-follow type. Possible types are:\n" \
                                   "All:     Perfectly follow emitter SRT. Particles after emission move in accordance with emitter transform animations.\n" \
                                   "Null:    Do not follow emitter. Particles after emission do not follow emitter transform animations.\n" \
                                   "PosOnly: Follow only emitter position. Particles after emission move in accordance with emitter position transforms only.",
                    "value": self.ptclFollowType.name
                },
                "randomSeed": {
                    "description": "Seed value used to initialize the Random Number Generator (RNG) of this emitter. Possible values are:\n" \
                                   "PerEmitter:      Seed is generated randomly per emitter.\n" \
                                   "PerSet:          Seed is generated randomly per emitter set. (Emitters using this type within the same set will use the same random seed.)\n" \
                                   "1 to 4294967294: Constant seed value to use directly. (This will cause the emitter to always behave the same every time it is spawned.)",
                    "value": 'PerEmitter' if self.randomSeed == 0 else ('PerSet' if self.randomSeed == 0xFFFFFFFF else self.randomSeed)
                },
                "drawParam": {
                    "isVisible": {
                        "description": "Draw the particles of this emitter? (This setting does not affect the child emitter, if present.)",
                        "value": self.isDisplayParent
                    },
                    "drawPath": {
                        "description": "Draw path index. (0-31)",
                        "value": self.drawPath
                    },
                    "particleSort": {
                        "description": "Order in which to draw particles. Possible orders are:\n" \
                                       "Ascending\n" \
                                       "Descending\n" \
                                       "ZSort",
                        "value": self.ptclDrawOrder.name
                    }
                }
            },
            "emitter": {
                "shape": {
                    "type": {
                        "description": "Shape type of the emitter. Possible types are:\n" \
                                       "Point:              Emits particles from a single fixed point. Each particle originates from the same position.\n" \
                                       "Circle:             Emits particles from random positions along the circumference of a circle/ellipse, or an arc.\n" \
                                       "CircleSameDivide:   Emits particles incrementally along the circumference of a circle/ellipse, or an arc, at equal spacing. (Number of partitions is controlled by emission rate.)\n" \
                                       "FillCircle:         Emits particles from random positions within the area of a circle/ellipse, or the area of a circular/elliptical sector defined by an arc.\n" \
                                       "Sphere:             Emits particles from random positions along the surface of a sphere/ellipsoid, or the surface of a spherical/ellipsoidal wedge defined by an arc either along the longitude or latitude.\n" \
                                       "SphereSameDivide:   Emits particles incrementally along the surface of a sphere/ellipsoid at equal spacing. (Number of partitions is specified by `tblIndex`.)\n" \
                                       "SphereSameDivide64: Same as `SphereSameDivide`, but has more partitions.\n" \
                                       "FillSphere:         Emits particles from random positions within the area of a sphere/ellipsoid, or the area of a spherical/ellipsoidal wedge defined by an arc either along the longitude or latitude.\n" \
                                       "Cylinder:           Emits particles from random positions along the surface of a(n elliptic) cylinder, or the surface of a(n elliptic) cylindrical sector defined by an arc.\n" \
                                       "FillCylinder:       Emits particles from random positions within the area of a(n elliptic) cylinder, or the area of a(n elliptic) cylindrical sector defined by an arc.\n" \
                                       "Box:                Emits particles from random positions along the surface of a box.\n" \
                                       "FillBox:            Emits particles from random positions within the area of a box.\n" \
                                       "Line:               Emits particles from random positions along a line.\n" \
                                       "LineSameDivide:     Emits particles incrementally along the a line at equal spacing. (Number of partitions is controlled by emission rate.)\n" \
                                       "Rectangle:          Emits particles from random positions along the sides of a rectangle.",
                        "value": self.volumeType.name
                    },
                    "extent": {
                        "description": "The extent of the shape. For each shape type, this means:\n" \
                                       "Point:     Value is not used.\n" \
                                       "*Circle*:  The lengths of the axes of the ellipse (or the radius of the circle if X = Z). Y axis is not used.\n" \
                                       "*Sphere*:  The lengths of the axes of the ellipsoid (or the radius of the sphere if X = Y = Z).\n" \
                                       "*Cylinder: The height (Y) of the cylinder, and the lengths (X & Z) of the axes of the elliptical cross-section (or the radius of the circular cross-section if X = Z).\n" \
                                       "*Box:      The width, height and length of the box.\n" \
                                       "Line*:     The length (Z) of the line. X & Y axes are not used.\n" \
                                       "Rectangle: The width and height of the rectangle. Y axis is not used.\n" \
                                       "Value must be positive.",
                        "value": Vec3ToDict(self.volumeRadius)
                    },
                    "extentScale": {
                        "description": "Scale to apply on top of shape extent. Value must be positive.",
                        "value": Vec3ToDict(self.formScale)
                    },
                    "tblIndex": {
                        "description": "For the two `SphereSameDivide*` shape types, used as an index to a table containing arrays of latitudes of equally-divided parts of a sphere surface. It determines possible latitudes the particle can be spawned at.\n" \
                                       "Value must be in the range:\n" \
                                       "SphereSameDivide:   [0, 7]\n" \
                                       "SphereSameDivide64: [0, 60]",
                        "value": self.volumeTblIndex
                    },
                    "caliber": {
                        "description": "Hollowness for `Fill` shape types. It specifies a region from which particles are not emitted as a percentage from the center of the emitter.\n" \
                                       "Value must be in the range `[0, 1]`, where a value of 1 is equivalent to 100%.",
                        "value": self.volumeCaliber
                    },
                    "lineCenter": {
                        "description": "Center position / half length for the two `Line*` shape types.",
                        "value": self.lineCenter
                    },
                    "arcOpening": {
                        "description": "For the `Sphere` and `FillSphere` shape types, method of arc opening. Possible values are:\n" \
                                       "Longitude: The arc is defined along the longitude. If this type is chosen, the values of `sweepParam` & `sweepStart`/`sweepStartRandom` are respected and the values of `latitude` & `latitudeDir` are ignored.\n" \
                                       "Latitude: The arc is defined along the latitude. If this type is chosen, the values of `latitude` & `latitudeDir` are respected and the values of `sweepParam` & `sweepStart`/`sweepStartRandom` are ignored.",
                        "value": self.isVolumeLatitudeEnabled.name
                    },
                    "sweepStart": {
                        "description": "For shape types which can use arcs, this is the arc start angle, in SIGNED 32-bit Binary Angular Measurement (BAM) where (+/-)0x40000000 = (+/-)90 degrees.",
                        "value": S32ToHexString(self.volumeSweepStart)
                    },
                    "sweepStartRandom": {
                        "description": "For shape types which can use arcs, randomly choose arc start angle? If True, the value of `sweepParam` is ignored.",
                        "value": self.volumeSweepStartRandom
                    },
                    "sweepParam": {
                        "description": "For shape types which can use arcs, this is the arc central angle, in UNSIGNED 32-bit Binary Angular Measurement (BAM) where 0x40000000 = 90 degrees. Positive values only!\n" \
                                       "Ignored if `sweepStartRandom` is True.",
                        "value": U32ToHexString(self.volumeSweepParam)
                    },
                    "latitude": {
                        "description": "For the `Sphere` and `FillSphere` shape types, minimum latitude at which the particle is spawned, in radians.\n" \
                                      f"Value must be in the range `[0, {MATH_PI}]`.",
                        "value": self.volumeLatitude
                    },
                    "latitudeDir": {
                        "description": "For the `Sphere` and `FillSphere` shape types, the basis to which the latitude should be transformed *to*. The original basis is +Y Axis.\n" \
                                       "Possible values are:\n" \
                                       "PosX: +X Axis.\n" \
                                       "NegX: -X Axis.\n" \
                                       "PosY: +Y Axis. (No transformation.)\n" \
                                       "NegY: -Y Axis.\n" \
                                       "PosZ: +Z Axis.\n" \
                                       "NegZ: -Z Axis.",
                        "value": self.volumeLatitudeDir.name
                    }
                },
                "transform": {
                    "scale": {
                        "description": "Emitter scale. Value must be positive.",
                        "value": Vec3ToDict(self.scale)
                    },
                    "rot": {
                        "description": "Emitter rotation, in radians.\n" \
                                      f"Value must be in the range `[0, {MATH_PI_2}]`.",
                        "value": Vec3ToDict(self.rot)
                    },
                    "rotRnd": {
                        "description": "Emitter rotation random range size, in radians.\n" \
                                       "A random decimal value picked from the range `(-rotRnd, rotRnd]` is added to `rot`.\n" \
                                      f"Value must be in the range `[0, {MATH_PI}]`.",
                        "value": Vec3ToDict(self.rotRnd)
                    },
                    "trans": {
                        "description": "Emitter translation.",
                        "value": Vec3ToDict(self.trans)
                    },
                    "transRnd": {
                        "description": "Emitter translation random range size.\n" \
                                       "A random decimal value picked from the range `(-transRnd, transRnd]` is added to `trans`.\n" \
                                       "Value must be positive.",
                        "value": Vec3ToDict(self.transRnd)
                    },
                    "mtxRT": {
                        "description": "Emitter RT transformation matrix. Optional.\n" \
                                       "If specified, will be used directly and will not be verified. Values must correspond to the specified `rot` and `trans`.\n" \
                                       "If omitted, will be calculated automatically from the specified `rot` and `trans`.",
                        "value": list(map(list, self.transformRT))
                    },
                    "mtxSRT": {
                        "description": "Emitter SRT transformation matrix. Optional.\n" \
                                       "If specified, will be used directly and will not be verified. Values must correspond to the specified `scale`, `rot` and `trans`.\n" \
                                       "If omitted, will be calculated automatically from the specified `scale`, `rot` and `trans`.",
                        "value": list(map(list, self.transformSRT))
                    }
                },
                "globalColor0": {
                    "description": "Emitter global multiplication color 0. Multiplies the particle color 0 RGB values with the specified RGB values.",
                    "value": Color3ToDict(self.color0)
                },
                "globalColor1": {
                    "description": "Emitter global multiplication color 1. Multiplies the particle color 1 RGB values with the specified RGB values.",
                    "value": Color3ToDict(self.color1)
                },
                "globalAlpha": {
                    "description": "Emitter global multiplication alpha. Multiplies the particle alpha value with the specified alpha value.\n" \
                                   "Value must be positive.",
                    "value": self.alpha
                }
            },
            "emission": {
                "timing": {
                    "startFrame": {
                        "description": "Emission start frame. Must be a positive integer value.",
                        "value": self.startFrame
                    },
                    "emitTime": {
                        "description": "Emission duration length. Possible values are:\n" \
                                       "Infinite: After the emission start frame is reached, the emitter continues emitting infinitely.\n" \
                                       "Positive integer value: The emission duration length in frames.",
                        "value": 'Infinite' if self.endFrame == SystemConstants.InfiniteLife else (self.endFrame - self.startFrame)
                    },
                    "interval": {
                        "type": {
                            "description": "Type of particle emission interval. Possible types are:\n" \
                                           "Time: Particles are emitted each time a certain amount of time passes.\n" \
                                           "Distance: Particles are emitted each time the emitter moves a certain unit of distance. (Equidistant particle emission.)",
                            "value": self.emitDistEnabled.name
                        },
                        "paramTime": {
                            "emitRate": {
                                "description": "Emission rate. This is the number of particles emitted per emission event.\n" \
                                               "Fractional values are allowed. For example, a value of of 1.5 would spawn 1 particle on some emission event, then 2 on the next, such that they add up to 3 particles per 2 emission events (3/2 = 1.5).\n" \
                                               "Value must be positive.",
                                "value": self.emitRate
                            },
                            "emitStep": {
                                "description": "Emission interval in frames. The number of frames between each emission event.\n" \
                                               "Value must be positive.",
                                "value": self.lifeStep
                            },
                            "emitStepRnd": {
                                "description": "Emission interval random addition range.\n" \
                                               "A random integer value picked from the range `[0, emitStepRnd)` is added to `emitStep`.\n" \
                                               "Value must be positive.",
                                "value": self.lifeStepRnd
                            }
                        },
                        "paramDistance": {
                            "emitDistUnit": {
                                "description": "Emission unit distance.\n" \
                                               "A particle is emitted when the sum of the distance per frame between the emitter position and the previous particle emission position exceeds this value.\n" \
                                               "Must be greater than zero.",
                                "value": self.emitDistUnit
                            },
                            "emitDistClampMax": {
                                "description": "Total travel distance per frame clamp range maximum value.\n" \
                                               "Value must be greater than or equal to `emitDistClampMin`.",
                                "value": self.emitDistMax
                            },
                            "emitDistClampMin": {
                                "description": "Total travel distance per frame clamp range minimum value.\n" \
                                               "Must be a positive value.",
                                "value": self.emitDistMin
                            },
                            "emitDistMargin": {
                                "description": "Total travel distance per frame truncation threshold.\n" \
                                               "If the distance in the current frame between the emitter position and the previous particle emission position is less than this value, the distance is truncated to 0.\n" \
                                               "This operation is applied before the distance is clamped within the range `[emitDistClampMin, emitDistClampMax]`.\n" \
                                               "Must be a positive value.",
                                "value": self.emitDistMargin
                            }
                        }
                    }
                },
                "posAndInitVel": {
                    "initPosRand": {
                        "description": "Particle initial position randomizer sphere radius.\n" \
                                       "If nonzero, the actual initial position of the particle ends up being a random point on the surface of a sphere centered on the original initial position of the particle. This value defines the radius of that sphere.\n" \
                                       "Value must be positive.",
                        "value": self.initPosRand
                    },
                    "figureVel": {
                        "description": "Magnitude of random-direction initial velocity.\n" \
                                       "This is the magnitude of the particle initial velocity in the direction of emission (and that direction is randomized to begin with).",
                        "value": self.figureVel
                    },
                    "emitterVel": {
                        "description": "Magnitude of emission velocity. Value must be positive.\n" \
                                       "This is the magnitude of a velocity added to the particle initial velocity at emission.",
                        "value": self.emitterVel
                    },
                    "emitterVelDir": {
                        "description": "Direction of emission velocity.\n" \
                                       "This is the direction of a velocity added to the particle initial velocity at emission.\n" \
                                       "Each axis value must be in the range `[-1, 1]`.",
                        "value": Vec3ToDict(self.emitterVelDir)
                    },
                    "emitterVelDirAngle": {
                        "description": "Dispersion angle of emission velocity, in degrees.\n" \
                                       "This value specifies the spread or cone angle within which the direction of emission velocity can vary. It controls how much the direction of the emission velocity can deviate its original direction.\n" \
                                       "Value must be in the range `[0.0, 180.0]`.\n" \
                                       "When the value is 0, the emission velocity is set directly to the direction specified by `emitterVelDir`, scaled by `emitterVel`.\n" \
                                       "When the value is greater than 0, the direction of emission velocity is randomly varied within a cone defined by this angle.\n" \
                                       "When the value is 180, the direction of emission velocity can vary within a full sphere centered around emitterVelDir.",
                        "value": self.emitterVelDirAngle
                    },
                    "xzDiffusionVel": {
                        "description": "Magnitude of initial horizontal (XZ) drift.\n" \
                                       "This is the magnitude of an additional velocity component added to the particle initial velocity in the horizontal (XZ) plane, perpendicular to the Y axis. It causes the particle to drift further in a direction influenced by its initial position in the XZ plane.",
                        "value": self.xzDiffusionVel
                    },
                    "spreadVec": {
                        "description": "Initial velocity spread vector. This attribute specifies the range of random variation applied to a particle initial velocity in the X, Y, and Z directions. It consists of three components:\n" \
                                       "* A random decimal value picked from the range `[-x, x)` is added to the X component of the particle initial velocity.\n" \
                                       "* A random decimal value picked from the range `[-y, y)` is added to the Y component of the particle initial velocity.\n" \
                                       "* A random decimal value picked from the range `[-z, z)` is added to the Z component of the particle initial velocity.\n" \
                                       "Values must be positive.\n" \
                                       "This spread vector is the *last* adjustment applied to the particle initial velocity. i.e., its effect is not impacted by `initVelRnd`.",
                        "value": Vec3ToDict(self.spreadVec)
                    },
                    "initVelRnd": {
                        "description": "Initial velocity random scale range. Value must be positive.\n" \
                                       "A random decimal value picked from the range `[1-initVelRnd, 1)` is used to scale the particle initial velocity.\n" \
                                       "This random scale is applied to the particle initial velocity *before* `spreadVec` is applied.",
                        "value": self.initVelRnd
                    },
                    "airRegist": {
                        "description": "Air resistance. This value is used to scale the particle velocity on every frame, and can be used to reduce or increase the particle velocity.\n" \
                                       "Value must be positive.",
                        "value": self.airRegist
                    }
                },
                "gravity": {
                    "isWorldGravity": {
                        "description": "Apply gravity in world coordinates?",
                        "value": self.isWorldGravity
                    },
                    "vector": {
                        "description": "Gravity vector. This value is added to the velocity and can also be thought of as the acceleration vector.",
                        "value": Vec3ToDict(self.gravity)
                    }
                }
            },
            "particle": {
                "lifespan": {
                    "description": "Particle lifespan. Possible values are:\n" \
                                   "Infinite: Once emitted, the particle lives infinitely.\n" \
                                   "Integer value greater than zero: The duration of the particle life from emission until expiration, in frames.",
                    "value": 'Infinite' if self.ptclLife == SystemConstants.InfiniteLife else self.ptclLife
                },
                "lifespanRnd": {
                    "description": "Particle lifespan random reduction range.\n" \
                                   "A random integer value picked from the range `[0, lifespanRnd)` is subtracted from `lifespan`.\n" \
                                   "Ignored if `lifespan` is set to `Infinite`.\n" \
                                   "Otherwise, value must be in the range `[0, lifespan]`.",
                    "value": self.ptclLifeRnd
                },
                "momentumRnd": {
                    "description": "Momentum random scale range. Value must be positive.\n" \
                                   "Makes individual particle motion appear random:\n" \
                                   "A random decimal value picked from the range `(1-momentumRnd, 1+momentumRnd]` is used to scale the particle velocity (as well as the Spin, Convergence and PosAdd fields).",
                    "value": self.dynamicsRandom
                },
                "shape": {
                    "type": {
                        "description": "Particle mesh type. Possible types are:\n" \
                                       "Particle:  The particle is a flat sheet.\n" \
                                       "Primitive: The particle uses a primitive model.\n" \
                                       "Stripe:    The particle is stretched out like fiber. This type is usually used with effects that leave a trail (stripe). `emitterType` must be set to `Complex` to be able to use this type.",
                        "value": self.meshType.name
                    },
                    "primitiveFigure": self.primitiveFigure.toDict(),
                    "billboardType": {
                        "description": "Billboard (particle orientation) type. Possible types are:\n" \
                                       "- If mesh type is `Particle` or `Primitive`, possible types are:\n" \
                                       "Billboard:          Normal billboard. The particle Y-axis is bound to the camera `up` vector and the Z-axis is always parallel to the camera lens axis.\n" \
                                       "YBillboard:         Y-axis billboard. Displays the view after rotating the particle around the Y-axis only, so that its Z-axis is parallel to the camera lens axis.\n" \
                                       "PolygonXY:          XY-plane polygon. This type of particle has sides in the X direction and the Y direction.\n" \
                                       "PolygonXZ:          XZ-plane polygon. This type of particle has sides in the X direction and the Z direction.\n" \
                                       "VelLook:            Directional Y-billboard. As this type of particle moves, it looks at its position in the previous frame and tilts in that direction of movement. It rotates only around the Y-axis to display with the Z-axis facing the camera.\n" \
                                       "VelLookPolygon:     Directional polygon. As this type of particle moves, it looks at its position in the previous frame and tilts in that direction of movement.\n" \
                                       "- If mesh type is `Stripe`, possible types are:\n" \
                                       "HistoricalStripe:   Historical Stripe. This type of particle is stretched out like a fiber and has a trail behind the particle.\n" \
                                       "ConsolidatedStripe: Consolidated Stripe. This type of particle is stretched out like a fiber and has a trail connected to particles in the emitter.",
                        "value": self.billboardType.name
                    },
                    "pivotOffset": {
                        "description": "Sets the offset of the pivot position for particle scaling and rotation, in local coordinates.",
                        "value": Vec2ToDict(self.rotBasis)
                    },
                    "toCameraOffset": {
                        "description": "Drawing offset in camera direction. Offsets the position where a particle is rendered either forward or backward relative to the camera.\n" \
                                       "Since this value is configurable per emitter, you can think of it as the Z-order of the emitter and can be used to adjust particle overlap between emitters.",
                        "value": self.toCameraOffset
                    }
                },
                "renderState": {
                    "blendType": {
                        "description": "Blend type. Possible types are:\n" \
                                       "Normal: Standard Blending, blends the source and destination colors using the source alpha value for smooth transparency effects, where the source partially covers the destination.\n" \
                                       "Add:    Additive Blending, adds the source color to the destination color, creating brightening effects such as glows or light flares.\n" \
                                       "Sub:    Subtractive Blending, subtracts the source color from the destination color using reverse subtraction, often used for creating inverted or darkening effects.\n" \
                                       "Screen: Screen Blending, combines the source and destination colors by inverting, multiplying, and adding them, useful for lightening the image and creating highlights.\n" \
                                       "Mult:   Multiplicative Blending, multiplies the source color with the destination color, commonly used for tinting, shading, or creating shadow effects.",
                        "value": self.blendType.name
                    },
                    "zBufATestType": {
                        "description": "Z-buffer alpha test type. Possible types are:\n" \
                                       "Normal:  Translucent Rendering (with depth testing), enables depth testing and sets the depth function to allow writing to fragments that are closer or at the same depth as existing ones, disables depth writing, and enables blending for proper rendering of transparent objects.\n" \
                                       "ZIgnore: Translucent Rendering (without depth testing), disables depth testing and depth writing, and enables blending, typically used for rendering effects that do not require depth sorting.\n" \
                                       "Entity:  Opaque Rendering (with depth and alpha testing), enables depth testing with depth writing, sets the depth function to allow writing to fragments that are closer or at the same depth as existing ones, uses alpha testing to discard fragments with alpha less than or equal to 0.5, and disables blending, making it suitable for rendering fully opaque objects that need depth sorting.",
                        "value": self.zBufATestType.name
                    },
                    "displaySide": {
                        "description": "Which side to display. Possible values are:\n" \
                                       "Both:  Display both sides of particles.\n" \
                                       "Front: Display only front side of particles.\n" \
                                       "Back:  Display only back side of particles.",
                        "value": self.displaySide.name
                    }
                }
            },
            "combiner": {
                "mode": {
                    "description": "Pixel combiner mode. Possible modes are:\n" \
                                   "- Normal:\n" \
                                   "In this mode, the new pixel output is calculated as follows:\n" \
                                   "PixelColor = Texture0Color;\n" \
                                   "if (Texture1IsPresent) PixelColor = Texture1ColorBlendFunc(PixelColor);\n" \
                                   "PixelColor = PtclColorBlendFunc(PixelColor);\n" \
                                   "if (MeshTypeIsPrimitive) PixelColor = PrimitiveColorBlendFunc(PixelColor);\n" \
                                   "PixelAlpha = Texture0Alpha;\n" \
                                   "if (Texture1IsPresent) PixelAlpha = Texture1AlphaBlendFunc(PixelAlpha);\n" \
                                   "if (MeshTypeIsPrimitive) PixelAlpha = PrimitiveAlphaBlendFunc(PixelAlpha);\n" \
                                   "PixelAlpha = PtclAlphaBlendFunc(PixelAlpha);\n" \
                                   "\n" \
                                   "- Refraction:\n" \
                                   "Color Buffer Refraction. Causes distortion of the background of the particle.\n" \
                                   "In this mode, the new pixel output is calculated as follows:\n" \
                                   "vec2 Offset = vec2(Texture0Color.r, Texture0Alpha) * offsetScale;\n" \
                                   "if (Texture1IsPresent && applyAlphaToRefract) Offset *= Texture1Alpha * PtclAlpha;\n" \
                                   "PixelColor = GetOriginalPixel(Offset);\n" \
                                   "PixelColor = PtclColorBlendFunc(PixelColor);\n" \
                                   "if (MeshTypeIsPrimitive) PixelColor = PrimitiveColorBlendFunc(PixelColor);\n" \
                                   "float AlphaTemp = 1.0;\n" \
                                   "if (MeshTypeIsPrimitive) AlphaTemp = PrimitiveAlphaBlendFunc(AlphaTemp);\n" \
                                   "PixelAlpha = PtclAlphaBlendFunc(AlphaTemp);\n" \
                                   "if (Texture1IsPresent) PixelAlpha *= Texture1Alpha * AlphaTemp;\n" \
                                   "\n" \
                                   "- Distortion:\n" \
                                   "Uses texture 0 to distort texture 1.\n" \
                                   "In this mode, the new pixel output is calculated as follows:\n" \
                                   "vec2 Offset = vec2(Texture0Color.r, Texture0Alpha) * offsetScale;\n" \
                                   "vec4 ColorTemp = GetTexture1Color(Offset);\n" \
                                   "float AlphaTemp = GetTexture1Alpha(Offset);\n" \
                                   "PixelColor = PtclColorBlendFunc(ColorTemp);\n" \
                                   "if (MeshTypeIsPrimitive) PixelColor = PrimitiveColorBlendFunc(PixelColor);\n" \
                                   "if (MeshTypeIsPrimitive) AlphaTemp = PrimitiveAlphaBlendFunc(AlphaTemp);\n" \
                                   "PixelAlpha = PtclAlphaBlendFunc(AlphaTemp);",
                    "value": self.shaderType.name
                },
                "offsetScale": {
                    "description": "Used to scale the offset in `Refraction` and `Distortion` modes.",
                    "value": Vec2ToDict((self.shaderParam0, self.shaderParam1))
                },
                "applyAlphaToRefract": {
                    "description": "Scale the offset using texture 1 & particle alpha in `Refraction` mode?",
                    "value": self.shaderApplyAlphaToRefract
                },
                "softParticle": {
                    "description": "This effect smooths out the edges of particles when they overlap with other objects in the scene, avoiding harsh intersections with surfaces. By enabling soft particles, you can make the particle system emit particles close to opaque surfaces without causing hard edges. Instead of having a hard, sharp edge, the particles gradually become more transparent near their boundaries, creating a softer transition between the particles and the background. This helps particles look less like flat images and more like volumetric objects.",
                    "enable": {
                        "description": "Enable soft particles?",
                        "value": self.shaderUseSoftEdge
                    },
                    "fadeDistance": {
                        "description": "Soft particle fade distance. This parameter sets how far from the point of overlap between the particle and the model the softening effect will start, i.e., the transition from opaque to transparent. A larger distance results in a more gradual fade.\n" \
                                       "Value must be positive.",
                        "value": self.softFadeDistance
                    },
                    "volume": {
                        "description": "Adjusts the perceived thickness of the particles. Higher values make the particles appear thicker by influencing how transparency is calculated based on the particle color brightness.\n" \
                                       "Value must be positive.",
                        "value": self.softVolumeParam
                    }
                },
                "colorCombiner": {
                    "texture0": {
                        "source": {
                            "description": "`Texture0Color` value source.\n" \
                                           "Possible sources are:\n" \
                                           "RGB: Color data of texture 0.\n" \
                                           "One: Constant value (1.0).",
                            "value": self.texture0ColorSource.name
                        }
                    },
                    "texture1": {
                        "source": {
                            "description": "`Texture1Color` value source.\n" \
                                           "Possible sources are:\n" \
                                           "RGB: Color data of texture 1.\n" \
                                           "One: Constant value (1.0).",
                            "value": self.texture1ColorSource.name
                        },
                        "colorBlendType": {
                            "description": "Type of color blending with `Texture1Color`.\n" \
                                           "Possible types are:\n" \
                                           "- Mod:\n" \
                                           "Texture1ColorBlendFunc(Color) = Color * Texture1Color;\n" \
                                           "- Add:\n" \
                                           "Texture1ColorBlendFunc(Color) = Color + Texture1Color;\n" \
                                           "- Sub:\n" \
                                           "Texture1ColorBlendFunc(Color) = Color - Texture1Color;",
                            "value": self.texture1ColorBlend.name
                        }
                    },
                    "ptclColor": {
                        "colorBlendType": {
                            "description": "Color calculation formula.\n" \
                                           "Possible types are:\n" \
                                           "- Color:\n" \
                                           "PtclColorBlendFunc(Color) = PtclColor0;\n" \
                                           "- Texture:\n" \
                                           "PtclColorBlendFunc(Color) = Color * PtclColor0;\n" \
                                           "- TextureInterpolate:\n" \
                                           "PtclColorBlendFunc(Color) = (Color * PtclColor0) + ((1 - Color) * PtclColor1);\n" \
                                           "- TextureAdd:\n" \
                                           "PtclColorBlendFunc(Color) = (Color * PtclColor0) + PtclColor1;",
                            "value": self.colorCombinerType.name
                        }
                    },
                    "primitive": {
                        "source": {
                            "description": "`PrimitiveColor` value source.\n" \
                                           "Possible sources are:\n" \
                                           "RGB: Color data of primitive.\n" \
                                           "One: Constant value (1.0).",
                            "value": self.primitiveColorSource.name
                        },
                        "colorBlendType": {
                            "description": "Type of color blending with `PrimitiveColor`.\n" \
                                           "Possible types are:\n" \
                                           "- Mod:\n" \
                                           "PrimitiveColorBlendFunc(Color) = Color * PrimitiveColor;\n" \
                                           "- Add:\n" \
                                           "PrimitiveColorBlendFunc(Color) = Color + PrimitiveColor;\n" \
                                           "- Sub:\n" \
                                           "PrimitiveColorBlendFunc(Color) = Color - PrimitiveColor;",
                            "value": self.primitiveColorBlend.name
                        }
                    }
                },
                "alphaCombiner": {
                    "commonSource": {
                        "description": "Common alpha source from texture 0, texture 1 and primitive.\n" \
                                       "Possible sources are:\n" \
                                       "Alpha: Alpha channel of texture 0, texture 1 and primitive color data.\n" \
                                       "Red:   Red channel of texture 0, texture 1 and primitive color data.",
                        "value": self.alphaCommonSource.name
                    },
                    "texture0": {
                        "source": {
                            "description": "`Texture0Alpha` value source.\n" \
                                           "Possible sources are:\n" \
                                           "Pass: Select from `commonSource`.\n" \
                                           "One:  Constant value (1.0).",
                            "value": self.texture0AlphaSource.name
                        }
                    },
                    "texture1": {
                        "source": {
                            "description": "`Texture1Alpha` value source.\n" \
                                           "Possible sources are:\n" \
                                           "Pass: Select from `commonSource`.\n" \
                                           "One:  Constant value (1.0).\n" \
                                           "If `Refraction` mode, the source is always assumed to be the alpha channel of texture 1 color data, regardless of what you set this to.",
                            "value": self.texture1AlphaSource.name
                        },
                        "alphaBlendType": {
                            "description": "Type of alpha blending with `Texture1Alpha`.\n" \
                                           "Possible types are:\n" \
                                           "- Mod:\n" \
                                           "Texture1AlphaBlendFunc(Alpha) = Alpha * Texture1Alpha;\n" \
                                           "- Add:\n" \
                                           "Texture1AlphaBlendFunc(Alpha) = Alpha + Texture1Alpha;\n" \
                                           "- Sub:\n" \
                                           "Texture1AlphaBlendFunc(Alpha) = Alpha - Texture1Alpha;",
                            "value": self.texture1AlphaBlend.name
                        }
                    },
                    "primitive": {
                        "source": {
                            "description": "`PrimitiveAlpha` value source.\n" \
                                           "Possible sources are:\n" \
                                           "Pass: Select from `commonSource`.\n" \
                                           "One:  Constant value (1.0).",
                            "value": self.primitiveAlphaSource.name
                        },
                        "alphaBlendType": {
                            "description": "Type of alpha blending with `PrimitiveAlpha`.\n" \
                                           "Possible types are:\n" \
                                           "- Mod:\n" \
                                           "PrimitiveAlphaBlendFunc(Alpha) = Alpha * PrimitiveAlpha;\n" \
                                           "- Add:\n" \
                                           "PrimitiveAlphaBlendFunc(Alpha) = Alpha + PrimitiveAlpha;\n" \
                                           "- Sub:\n" \
                                           "PrimitiveAlphaBlendFunc(Alpha) = Alpha - PrimitiveAlpha;",
                            "value": self.primitiveAlphaBlend.name
                        }
                    },
                    "ptclAlpha": {
                        "alphaBlendType": {
                            "description": "Alpha calculation formula.\n" \
                                           "Possible types are:\n" \
                                           "Mod:\n" \
                                           "PtclAlphaBlendFunc(Alpha) = Alpha * PtclAlpha;\n" \
                                           "Sub:\n" \
                                           "PtclAlphaBlendFunc(Alpha) = (Alpha - (1 - PtclAlpha)) * 2;",
                            "value": self.alphaBaseCombinerType.name
                        }
                    }
                }
            },
            **{f"texture{i}": TextureToDict(texRes, textureData) for i, (texRes, textureData) in enumerate(zip(self.texRes, self.textureData))},
            "ptclColor": {
                "colorScale": {
                    "description": "Color scale. Used to scale the final color 0 & color 1 values.\n" \
                                   "Value must be positive.",
                    "value": self.colorScale
                },
                **{f"color{i}": {
                    "colorCalcType": {
                        "description": "Color behavior type. Possible types are:\n" \
                                       "Fixed:     The value of this color attribute is fixed.\n" \
                                       "Random:    The value of this color attribute is randomly picked once then fixed.\n" \
                                       "Animation: The value of this color attribute is animated using 3 color elements in 4 sections that are fitted to the particle lifespan.",
                        "value": colorCalcType.name
                    },
                    "colorElem": {
                        "description": "Three RGB color elements. They have different significance based on the `colorCalcType` value:\n" \
                                       "Fixed:     The first element in this array is always used as the color.\n" \
                                       "Random:    One of these three elements is randomly picked as the color for each particle.\n" \
                                       "Animation: These three elements are used to animate the color.\n" \
                                       "Values must be positive.",
                        "value": [Color3ToDict(elem) for elem in color]
                    },
                    "animationParam": {
                        "colorSection1": {
                            "description": "End of 1st color section as percentage of particle lifespan.\n" \
                                           "Value must be in the range `[0, 100]`.\n" \
                                           "During this section, the color value is fixed to `colorElem[0]`.",
                            "value": colorSection1
                        },
                        "colorSection2": {
                            "description": "End of 2nd color section as percentage of particle lifespan.\n" \
                                           "Value must be in the range `[0, 100]`.\n" \
                                           "During this section, the color value transitions from `colorElem[0]` to `colorElem[1]` (using linear interpolation).",
                            "value": colorSection2
                        },
                        "colorSection3": {
                            "description": "End of 3rd color section as percentage of particle lifespan.\n" \
                                           "Value must be in the range `[0, 100]`.\n" \
                                           "During this section, the color value transitions from `colorElem[1]` to `colorElem[2]` (using linear interpolation).\n" \
                                           "(In the 4th color section (which is last), the color value is fixed to `colorElem[2]`.)",
                            "value": colorSection3
                        },
                        "colorNumRepeat": {
                            "description": "Color animation repetition count.\n" \
                                           "Specifies how many times this color animation must be played over the particle lifespan.\n" \
                                           "Must be greater than zero.",
                            "value": colorNumRepeat
                        },
                        "colorRepeatStartRand": {
                            "description": "Set color animation start position randomly (per particle)?\n" \
                                           "In other words, if enabled, instead of the color animation starting from the beginning of the 1st color section, the animation will start from a random point in one of the 4 sections (for each particle).",
                            "value": colorRepeatStartRand
                        }
                    }
                } for i, (colorCalcType, color, colorSection1, colorSection2, colorSection3, colorNumRepeat, colorRepeatStartRand) in enumerate(zip(self.colorCalcType, self.color, self.colorSection1, self.colorSection2, self.colorSection3, self.colorNumRepeat, self.colorRepeatStartRand))}
            },
            "ptclAlpha": {
                "description": "The value of the particle alpha attribute is animated using 3 alpha elements in 3 sections that are fitted to the particle lifespan. If `alphaSection1` is set to 0 AND `alphaSection2` is set to 100, the alpha animation is disabled.",
                "animationParam": {
                    "alphaElem[0]": {
                        "description": "Alpha animation element 0.\n" \
                                       "Value must be in the range `[0, 1]`",
                        "value": self.initAlpha
                    },
                    "diffAlphaElem10": {
                        "description": "The difference between alpha elements 0 & 1.\n" \
                                       "(`alphaElem[1] = alphaElem[0] + diffAlphaElem10`)\n" \
                                       "Value must be such that `alphaElem[1]` is in the range `[0, 1]`.",
                        "value": self.diffAlpha21
                    },
                    "diffAlphaElem21": {
                        "description": "The difference between the alpha elements 1 & 2.\n" \
                                       "(`alphaElem[2] = alphaElem[1] + diffAlphaElem21`)\n" \
                                       "Value must be such that `alphaElem[2]` is in the range `[0, 1]`.",
                        "value": self.diffAlpha32
                    },
                    "alphaSection1": {
                        "description": "End of 1st alpha section as percentage of particle lifespan.\n" \
                                       "Value must be in the range `[0, 100]`.\n" \
                                       "During this section, the alpha value transitions from `alphaElem[0]` to `alphaElem[1]` (using linear interpolation).",
                        "value": self.alphaSection1
                    },
                    "alphaSection2": {
                        "description": "End of 2nd alpha section as percentage of particle lifespan.\n" \
                                       "Value must be in the range `[0, 100]`.\n" \
                                       "During this section, the alpha value is fixed to `alphaElem[1]`.\n" \
                                       "(In the 3rd alpha section (which is last), the alpha value transitions from `alphaElem[1]` to `alphaElem[2]` (using linear interpolation).)",
                        "value": self.alphaSection2
                    }
                }
            },
            "ptclScale": {
                "description": "The value of the particle scale attribute is animated using 3 scale percentage elements in 3 sections that are fitted to the particle lifespan.\n" \
                               "At every point in time, the particle scale is set to the particle scale base value (`baseScale`) multiplied by the current scale percentage.\n" \
                               "For all scale percentage values, a value of 1 is equivalent to 100%.\n" \
                               "If `scaleSection1` is set to -127 AND `scaleSection2` is set to 100, the scale animation is disabled.",
                "baseScale": {
                    "description": "Particle Scale base value. Must be positive.",
                    "value": Vec2ToDict(self.baseScale)
                },
                "baseScaleRand": {
                    "description": "Particle Scale base value random percentage range.\n" \
                                   "A random decimal value picked from the range `(1-baseScaleRand, 1]` is used to scale the particle scale base value (`baseScale`).\n" \
                                   "Value must be positive.",
                    "value": self.scaleRand
                },
                "animationParam": {
                    "scaleElem[0]": {
                        "description": "Scale percentage animation element 0.",
                        "value": Vec2ToDict(self.initScale)
                    },
                    "diffScaleElem10": {
                        "description": "The difference between scale percentage elements 0 & 1.\n" \
                                       "(`scaleElem[1] = scaleElem[0] + diffScaleElem10`)",
                        "value": Vec2ToDict(self.diffScale21)
                    },
                    "diffScaleElem21": {
                        "description": "The difference between the scale percentage elements 1 & 2.\n" \
                                       "(`scaleElem[2] = scaleElem[1] + diffScaleElem21`)",
                        "value": Vec2ToDict(self.diffScale32)
                    },
                    "scaleSection1": {
                        "description": "End of 1st scale section as percentage of particle lifespan.\n" \
                                       "Value must either be in the range `[0, 100]`, or be -127.\n" \
                                       "During this section, the scale percentage value transitions from `scaleElem[0]` to `scaleElem[1]` (using linear interpolation).",
                        "value": self.scaleSection1
                    },
                    "scaleSection2": {
                        "description": "End of 2nd scale section as percentage of particle lifespan.\n" \
                                       "Value must be in the range `[0, 100]`.\n" \
                                       "During this section, the scale percentage value is fixed to `scaleElem[1]`.\n" \
                                       "(In the 3rd scale section (which is last), the scale percentage value transitions from `scaleElem[1]` to `scaleElem[2]` (using linear interpolation).)",
                        "value": self.scaleSection2
                    }
                }
            },
            "ptclRot": {
                "type": {
                    "description": "Particle rotation type. Possible types are:\n" \
                                   "NoWork: Particles do not rotate.\n" \
                                   "RotX:   Particles can rotate only around on the X axis.\n" \
                                   "RotY:   Particles can rotate only around on the Y axis.\n" \
                                   "RotZ:   Particles can rotate only around on the Z axis.\n" \
                                   "RotXYZ: Particles can rotate around all axes.",
                    "value": self.ptclRotType.name
                },
                "initRot": {
                    "description": "Particle initial rotation, in radians.\n" \
                                  f"Value must be in the range `[0, {MATH_PI_2}]`.",
                    "value": Vec3ToDict(self.initRot)
                },
                "initRotRand": {
                    "description": "Particle initial rotation random range size, in radians.\n" \
                                   "A random decimal value picked from the range `[0, initRotRand)` is added to `initRot`.\n" \
                                  f"Value must be in the range `[0, {MATH_PI_2}]`.",
                    "value": Vec3ToDict(self.initRotRand)
                },
                "rotRegist": {
                    "description": "Particle rotation air resistance. This value is used to scale the particle angular velocity on every frame, and can be used to reduce or increase the particle angular velocity.\n" \
                                   "Value must be positive.",
                    "value": self.rotRegist
                },
                "rotVel": {
                    "description": "Particle angular velocity, in radians.\n" \
                                  f"Value must be in the range `[{-MATH_PI}, {MATH_PI}]`.",
                    "value": Vec3ToDict(self.rotVel)
                },
                "rotVelRand": {
                    "description": "Particle angular velocity random range size, in radians.\n" \
                                   "A random decimal value picked from the range `[0, rotVelRand)` is added to `rotVel`.\n" \
                                  f"Value must be in the range `[0, {MATH_PI_2}]`.",
                    "value": Vec3ToDict(self.rotVelRand)
                }
            },
            "termination": {
                "isStopEmitInFade": {
                    "description": "Cease particle emission during fade out?",
                    "value": self.isStopEmitInFade
                },
                "alphaAddInFade": {
                    "description": "Alpha fade rate. This value will be subtracted from/added to the particle fading alpha multiplier during fade out/in.\n" \
                                   "Value must be positive.",
                    "value": self.alphaAddInFade
                }
            },
            "userData": {
                "callbackID": {
                    "description": "User data callback ID (0-7 or null).",
                    "value": None if self.userCallbackID == UserDataCallBackID.Null else self.userCallbackID.value
                },
                "bitfield": {
                    "description": "16-bit bitfield.",
                    "value": f'0b{self.userDataBit:016b}'
                },
                "numArrayU8": {
                    "description": "1-byte number array (Size = 6).",
                    "value": list(self.userDataU8)
                },
                "numArrayF": {
                    "description": f"Decimal number array (Size = {UserDataParamIdx.Max}).",
                    "value": list(self.userDataF)
                }
            },
            "userShader": {
                "shaderType": {
                    "description": "Games and applications are allowed to select a global shader type (ref. `nw::eft::Renderer::SetShaderType` function) that define special environments under which user shaders are allowed to behave differently. Three global types exist (0, 1, 2). Type 0 is the type under which user shaders should behave as normal. Types 1 and 2 are special types that user shaders can react to.",
                    "macroDef1": {
                        "description": "A macro to be dynamically defined at shader compilation when the global shader type is set to 1. The user shader can detect if the global type is set to 1 by checking if this macro has been defined. Value must be encodeable to less than 16 bytes using Shift JIS.",
                        "value": self.userShaderDefine1 if self.userShaderDefine1 else None
                    },
                    "macroDef2": {
                        "description": "A macro to be dynamically defined at shader compilation when the global shader type is set to 2. The user shader can detect if the global type is set to 2 by checking if this macro has been defined. Value must be encodeable to less than 16 bytes using Shift JIS.",
                        "value": self.userShaderDefine2 if self.userShaderDefine2 else None
                    }
                },
                "localType": {
                    "description": "Local shader type that can be selected per emitter, in contrast to the global shader type. Nine local types exist (0-8). Type 0 is the type under which user shaders should behave as normal. Types 1 to 8 are special types that user shaders can react to.\n" \
                                   "- If the value is set to 0, the macros `USR_SETTING_NONE`, `USR_VERTEX_SETTING_NONE` and `USR_FRAGMENT_SETTING_NONE` are dynamically defined at shader compilation.\n" \
                                   "- If the value is set to some number `X` between 1 and 8, the macros `USR_SETTING_X`, `USR_VERTEX_SETTING_X` and `USR_FRAGMENT_SETTING_X` are dynamically defined at shader compilation.",
                    "value": self.userShaderSetting
                },
                "bitfield": {
                    "description": "A 32-bit bitfield specifying 32 flags which are possible to combine.\n" \
                                   "For each bit X between 0 and 31 that is set to 1, the macros `USR_FLAG_X`, `USR_VERTEX_FLAG_X` and `USR_FRAGMENT_FLAG_X` are dynamically defined at shader compilation.",
                    "value": f'0b{self.userShaderFlag:032b}'
                },
                "switchCase": {
                    "description": "A 32-bit bitfield specifying 32 switch cases.\n" \
                                   "For each bit X between 0 and 31 that is set to 1, the macros `USR_SWITCH_FLAG_X`, `USR_VERTEX_SWITCH_FLAG_X` and `USR_FRAGMENT_SWITCH_FLAG_X` are dynamically defined at shader compilation.",
                    "value": f'0b{self.userShaderSwitchFlag:032b}'
                },
                "param": self.userShaderParam.toDict()
            },
            "complex": {
                "description": "Options that are only available when the emitter type is set to `Complex`.",
                "field": {
                    "random": fieldRandomData.toDict(fieldRandomEnable),
                    "magnet": fieldMagnetData.toDict(fieldMagnetEnable),
                    "spin": fieldSpinData.toDict(fieldSpinEnable),
                    "convergence": fieldConvergenceData.toDict(fieldConvergenceEnable),
                    "posAdd": fieldPosAddData.toDict(fieldPosAddEnable),
                    "collision": fieldCollisionData.toDict(fieldCollisionEnable)
                },
                "flux": fluctuationData.toDict(fluctuationEnable, fluctuationApplyAlpha, fluctuationApplyScale),
                "child": childData.toDict(childEnable),
                "stripe": stripeData.toDict(stripeEmitterCoord)
            },
            "unused": {
                "description": "Attributes which are stored in the emitter data, but not actually used in the code, and therefore they are useless.",
                "isPolygon": {
                    "description": "Is polygon?",
                    "value": self.isPolygon
                },
                "isFollowAll": {
                    "value": self.isFollowAll
                },
                "isEmitterBillboardMtx": {
                    "description": "Can the billboard matrix be set per emitter?",
                    "value": self.isEmitterBillboardMtx
                },
                "isDirectional": {
                    "value": self.isDirectional
                }
            }
        }

    def fromDict(self, dic: DictGeneric) -> None:
        dic_general: DictGeneric = dic["general"]
        dic_general_drawParam: DictGeneric = dic_general["drawParam"]
        dic_emitter: DictGeneric = dic["emitter"]
        dic_emitter_shape: DictGeneric = dic_emitter["shape"]
        dic_emitter_transform: DictGeneric = dic_emitter["transform"]
        dic_emission: DictGeneric = dic["emission"]
        dic_emission_timing: DictGeneric = dic_emission["timing"]
        dic_emission_timing_interval: DictGeneric = dic_emission_timing["interval"]
        dic_emission_timing_interval_paramTime: DictGeneric = dic_emission_timing_interval["paramTime"]
        dic_emission_timing_interval_paramDistance: DictGeneric = dic_emission_timing_interval["paramDistance"]
        dic_emission_posAndInitVel: DictGeneric = dic_emission["posAndInitVel"]
        dic_emission_gravity: DictGeneric = dic_emission["gravity"]
        dic_particle: DictGeneric = dic["particle"]
        dic_particle_shape: DictGeneric = dic_particle["shape"]
        dic_particle_renderState: DictGeneric = dic_particle["renderState"]
        dic_combiner: DictGeneric = dic["combiner"]
        dic_combiner_softParticle: DictGeneric = dic_combiner["softParticle"]
        dic_combiner_colorCombiner: DictGeneric = dic_combiner["colorCombiner"]
        dic_combiner_colorCombiner_texture1: DictGeneric = dic_combiner_colorCombiner["texture1"]
        dic_combiner_colorCombiner_primitive: DictGeneric = dic_combiner_colorCombiner["primitive"]
        dic_combiner_alphaCombiner: DictGeneric = dic_combiner["alphaCombiner"]
        dic_combiner_alphaCombiner_texture1: DictGeneric = dic_combiner_alphaCombiner["texture1"]
        dic_combiner_alphaCombiner_primitive: DictGeneric = dic_combiner_alphaCombiner["primitive"]
        dic_ptclColor: DictGeneric = dic["ptclColor"]
        dic_ptclAlpha_animationParam: DictGeneric = dic["ptclAlpha"]["animationParam"]
        dic_ptclScale: DictGeneric = dic["ptclScale"]
        dic_ptclScale_animationParam: DictGeneric = dic_ptclScale["animationParam"]
        dic_ptclRot: DictGeneric = dic["ptclRot"]
        dic_termination: DictGeneric = dic["termination"]
        dic_userData: DictGeneric = dic["userData"]
        dic_userShader: DictGeneric = dic["userShader"]
        dic_userShader_shaderType: DictGeneric = dic_userShader["shaderType"]
        dic_complex: DictGeneric = dic["complex"]
        dic_complex_field: DictGeneric = dic_complex["field"]
        dic_unused: DictGeneric = dic["unused"]

        self.type = EmitterType[dic_general["emitterType"]["value"]]
        self.ptclFollowType = PtclFollowType[dic_general["ptclFollowType"]["value"]]
        randomSeed = dic_general["randomSeed"]["value"]
        if randomSeed == 'PerEmitter':
            self.randomSeed = 0
        elif randomSeed == 'PerSet':
            self.randomSeed = 0xFFFFFFFF
        else:
            self.randomSeed = VerifyIntRange(randomSeed, 1, 4294967294)
        self.isDisplayParent = VerifyBool(dic_general_drawParam["isVisible"]["value"])
        self.drawPath = VerifyIntRange(dic_general_drawParam["drawPath"]["value"], 0, 31)
        self.ptclDrawOrder = ParticleDrawOrder[dic_general_drawParam["particleSort"]["value"]]

        self.volumeType = VolumeType[dic_emitter_shape["type"]["value"]]
        self.volumeRadius = Vec3PositiveFromDict(dic_emitter_shape["extent"]["value"])
        self.formScale = Vec3PositiveFromDict(dic_emitter_shape["extentScale"]["value"])
        if self.volumeType == VolumeType.SphereSameDivide:
            self.volumeTblIndex = VerifyIntRange(dic_emitter_shape["tblIndex"]["value"], 0, 7)
        elif self.volumeType == VolumeType.SphereSameDivide64:
            self.volumeTblIndex = VerifyIntRange(dic_emitter_shape["tblIndex"]["value"], 0, 60)
        else:
            self.volumeTblIndex = VerifyU8(dic_emitter_shape["tblIndex"]["value"])
        self.volumeCaliber = VerifyF32Normal(dic_emitter_shape["caliber"]["value"])
        self.lineCenter = VerifyF32(dic_emitter_shape["lineCenter"]["value"])
        self.isVolumeLatitudeEnabled = ArcOpeningType[dic_emitter_shape["arcOpening"]["value"]]
        self.volumeSweepStart = VerifyS32(int(dic_emitter_shape["sweepStart"]["value"], 16))
        self.volumeSweepStartRandom = VerifyBool(dic_emitter_shape["sweepStartRandom"]["value"])
        self.volumeSweepParam = VerifyU32(int(dic_emitter_shape["sweepParam"]["value"], 16))
        self.volumeLatitude = VerifyF32Range(dic_emitter_shape["latitude"]["value"], 0.0, MATH_PI)
        self.volumeLatitudeDir = VolumeLatitudeDir[dic_emitter_shape["latitudeDir"]["value"]]
        self.scale = Vec3PositiveFromDict(dic_emitter_transform["scale"]["value"])
        self.rot = Vec3RangeFromDict(dic_emitter_transform["rot"]["value"], 0.0, MATH_PI_2)
        self.rotRnd = Vec3RangeFromDict(dic_emitter_transform["rotRnd"]["value"], 0.0, MATH_PI)
        self.trans = Vec3FromDict(dic_emitter_transform["trans"]["value"])
        self.transRnd = Vec3PositiveFromDict(dic_emitter_transform["transRnd"]["value"])
        if "mtxRT" in dic_emitter_transform:
            transformRT = dic_emitter_transform["mtxRT"]["value"]
            assert isinstance(transformRT, list) and len(transformRT) == 3
            assert all((isinstance(row, list) and len(row) == 4) for row in transformRT)
            self.transformRT = tuple(tuple(VerifyF32(col) for col in row) for row in transformRT)
        else:
            self.transformRT = Mtx34MakeRT(self.rot, self.trans)
        if "mtxSRT" in dic_emitter_transform:
            transformSRT = dic_emitter_transform["mtxSRT"]["value"]
            assert isinstance(transformSRT, list) and len(transformSRT) == 3
            assert all((isinstance(row, list) and len(row) == 4) for row in transformSRT)
            self.transformSRT = tuple(tuple(VerifyF32(col) for col in row) for row in transformSRT)
        else:
            self.transformSRT = Mtx34MakeSRT(self.scale, self.rot, self.trans)
        self.color0 = Color3PositiveFromDict(dic_emitter["globalColor0"]["value"])
        self.color1 = Color3PositiveFromDict(dic_emitter["globalColor1"]["value"])
        self.alpha = VerifyF32Positive(dic_emitter["globalAlpha"]["value"])

        self.startFrame = VerifyS32Positive(dic_emission_timing["startFrame"]["value"])
        emitTime = dic_emission_timing["emitTime"]["value"]
        if emitTime == 'Infinite':
            self.endFrame = int(SystemConstants.InfiniteLife)
        else:
            self.endFrame = self.startFrame + VerifyIntRange(emitTime, 0, 0x7FFFFFFF - self.startFrame)
        self.emitDistEnabled = EmissionIntervalType[dic_emission_timing_interval["type"]["value"]]
        self.emitRate = VerifyF32Positive(dic_emission_timing_interval_paramTime["emitRate"]["value"])
        self.lifeStep = VerifyS32Positive(dic_emission_timing_interval_paramTime["emitStep"]["value"])
        self.lifeStepRnd = VerifyS32Positive(dic_emission_timing_interval_paramTime["emitStepRnd"]["value"])
        self.emitDistUnit = VerifyF32PositiveNonZero(dic_emission_timing_interval_paramDistance["emitDistUnit"]["value"])
        self.emitDistMax = VerifyF32(dic_emission_timing_interval_paramDistance["emitDistClampMax"]["value"])
        self.emitDistMin = VerifyF32Range(dic_emission_timing_interval_paramDistance["emitDistClampMin"]["value"], 0, self.emitDistMax)
        self.emitDistMargin = VerifyF32Positive(dic_emission_timing_interval_paramDistance["emitDistMargin"]["value"])
        self.initPosRand = VerifyF32Positive(dic_emission_posAndInitVel["initPosRand"]["value"])
        self.figureVel = VerifyF32(dic_emission_posAndInitVel["figureVel"]["value"])
        self.emitterVel = VerifyF32Positive(dic_emission_posAndInitVel["emitterVel"]["value"])
        self.emitterVelDir = Vec3RangeFromDict(dic_emission_posAndInitVel["emitterVelDir"]["value"], -1.0, 1.0)
        self.emitterVelDirAngle = VerifyF32Range(dic_emission_posAndInitVel["emitterVelDirAngle"]["value"], 0.0, 180.0)
        self.xzDiffusionVel = VerifyF32(dic_emission_posAndInitVel["xzDiffusionVel"]["value"])
        self.spreadVec = Vec3PositiveFromDict(dic_emission_posAndInitVel["spreadVec"]["value"])
        self.initVelRnd = VerifyF32Positive(dic_emission_posAndInitVel["initVelRnd"]["value"])
        self.airRegist = VerifyF32Positive(dic_emission_posAndInitVel["airRegist"]["value"])
        self.isWorldGravity = VerifyBool(dic_emission_gravity["isWorldGravity"]["value"])
        self.gravity = Vec3FromDict(dic_emission_gravity["vector"]["value"])

        lifespan = dic_particle["lifespan"]["value"]
        if lifespan == 'Infinite':
            self.ptclLife = int(SystemConstants.InfiniteLife)
        else:
            self.ptclLife = VerifyS32PositiveNonZero(lifespan)
        self.ptclLifeRnd = VerifyIntRange(dic_particle["lifespanRnd"]["value"], 0, self.ptclLife)
        self.dynamicsRandom = VerifyF32Positive(dic_particle["momentumRnd"]["value"])
        self.meshType = MeshType[dic_particle_shape["type"]["value"]]
        self.primitiveFigure.fromDict(dic_particle_shape["primitiveFigure"])
        self.billboardType = BillboardType[dic_particle_shape["billboardType"]["value"]]
        self.rotBasis = Vec2FromDict(dic_particle_shape["pivotOffset"]["value"])
        self.toCameraOffset = VerifyF32(dic_particle_shape["toCameraOffset"]["value"])
        self.blendType = BlendType[dic_particle_renderState["blendType"]["value"]]
        self.zBufATestType = ZBufATestType[dic_particle_renderState["zBufATestType"]["value"]]
        self.displaySide = DisplaySideType[dic_particle_renderState["displaySide"]["value"]]

        self.shaderType = FragmentShaderVariation[dic_combiner["mode"]["value"]]
        self.shaderParam0, self.shaderParam1 = Vec2PositiveFromDict(dic_combiner["offsetScale"]["value"])
        self.shaderApplyAlphaToRefract = VerifyBool(dic_combiner["applyAlphaToRefract"]["value"])
        self.shaderUseSoftEdge = VerifyBool(dic_combiner_softParticle["enable"]["value"])
        self.softFadeDistance = VerifyF32Positive(dic_combiner_softParticle["fadeDistance"]["value"])
        self.softVolumeParam = VerifyF32Positive(dic_combiner_softParticle["volume"]["value"])
        self.texture0ColorSource = ColorSource[dic_combiner_colorCombiner["texture0"]["source"]["value"]]
        self.texture1ColorSource = ColorSource[dic_combiner_colorCombiner_texture1["source"]["value"]]
        self.texture1ColorBlend = ColorAlphaBlendType[dic_combiner_colorCombiner_texture1["colorBlendType"]["value"]]
        self.colorCombinerType = ColorCombinerType[dic_combiner_colorCombiner["ptclColor"]["colorBlendType"]["value"]]
        self.primitiveColorSource = ColorSource[dic_combiner_colorCombiner_primitive["source"]["value"]]
        self.primitiveColorBlend = ColorAlphaBlendType[dic_combiner_colorCombiner_primitive["colorBlendType"]["value"]]
        self.alphaCommonSource = AlphaCommonSource[dic_combiner_alphaCombiner["commonSource"]["value"]]
        self.texture0AlphaSource = AlphaSource[dic_combiner_alphaCombiner["texture0"]["source"]["value"]]
        self.texture1AlphaSource = AlphaSource[dic_combiner_alphaCombiner_texture1["source"]["value"]]
        self.texture1AlphaBlend = ColorAlphaBlendType[dic_combiner_alphaCombiner_texture1["alphaBlendType"]["value"]]
        self.primitiveAlphaSource = AlphaSource[dic_combiner_alphaCombiner_primitive["source"]["value"]]
        self.primitiveAlphaBlend = ColorAlphaBlendType[dic_combiner_alphaCombiner_primitive["alphaBlendType"]["value"]]
        self.alphaBaseCombinerType = AlphaBaseCombinerType[dic_combiner_alphaCombiner["ptclAlpha"]["alphaBlendType"]["value"]]

        for i, (texRes, textureData) in enumerate(zip(self.texRes, self.textureData)):
            TextureFromDict(texRes, textureData, dic[f"texture{i}"])

        self.colorScale = VerifyF32Positive(dic_ptclColor["colorScale"]["value"])
        colorCalcType: List[ColorCalcType] = []
        color: List[Tuple[Color3Numeric, Color3Numeric, Color3Numeric]] = []
        colorSection1: List[int] = []
        colorSection2: List[int] = []
        colorSection3: List[int] = []
        colorNumRepeat: List[int] = []
        colorRepeatStartRand: List[bool] = []
        for i in range(ColorKind.Max):
            dic_ptclColor_color: DictGeneric = dic_ptclColor[f"color{i}"]
            dic_ptclColor_color_animationParam: DictGeneric = dic_ptclColor_color["animationParam"]
            colorCalcType.append(ColorCalcType[dic_ptclColor_color["colorCalcType"]["value"]])
            colorElem = dic_ptclColor_color["colorElem"]["value"]
            assert isinstance(colorElem, list) and len(colorElem) == 3
            color.append(tuple(Color3PositiveFromDict(elem) for elem in colorElem))
            colorSection1.append(VerifyIntRange(dic_ptclColor_color_animationParam["colorSection1"]["value"], 0, 100))
            colorSection2.append(VerifyIntRange(dic_ptclColor_color_animationParam["colorSection2"]["value"], 0, 100))
            colorSection3.append(VerifyIntRange(dic_ptclColor_color_animationParam["colorSection3"]["value"], 0, 100))
            colorNumRepeat.append(VerifyS32PositiveNonZero(dic_ptclColor_color_animationParam["colorNumRepeat"]["value"]))
            colorRepeatStartRand.append(VerifyBool(dic_ptclColor_color_animationParam["colorRepeatStartRand"]["value"]))
        self.colorCalcType = tuple(colorCalcType)
        self.color = tuple(color)
        self.colorSection1 = tuple(colorSection1)
        self.colorSection2 = tuple(colorSection2)
        self.colorSection3 = tuple(colorSection3)
        self.colorNumRepeat = tuple(colorNumRepeat)
        self.colorRepeatStartRand = tuple(colorRepeatStartRand)

        self.initAlpha = VerifyF32Normal(dic_ptclAlpha_animationParam["alphaElem[0]"]["value"])
        self.diffAlpha21 = VerifyF32(dic_ptclAlpha_animationParam["diffAlphaElem10"]["value"])
        self.diffAlpha32 = VerifyF32(dic_ptclAlpha_animationParam["diffAlphaElem21"]["value"])
        alpha2: float = self.initAlpha + self.diffAlpha21
        alpha3: float = alpha2 + self.diffAlpha32
        assert 0.0 <= alpha2 <= 1.0
        assert 0.0 <= alpha3 <= 1.0
        self.alphaSection1 = VerifyIntRange(dic_ptclAlpha_animationParam["alphaSection1"]["value"], 0, 100)
        self.alphaSection2 = VerifyIntRange(dic_ptclAlpha_animationParam["alphaSection2"]["value"], 0, 100)

        self.baseScale = Vec2PositiveFromDict(dic_ptclScale["baseScale"]["value"])
        self.scaleRand = VerifyF32Positive(dic_ptclScale["baseScaleRand"]["value"])
        self.initScale = Vec2FromDict(dic_ptclScale_animationParam["scaleElem[0]"]["value"])
        self.diffScale21 = Vec2FromDict(dic_ptclScale_animationParam["diffScaleElem10"]["value"])
        self.diffScale32 = Vec2FromDict(dic_ptclScale_animationParam["diffScaleElem21"]["value"])
        scaleSection1 = dic_ptclScale_animationParam["scaleSection1"]["value"]
        if scaleSection1 == -127:
            self.scaleSection1 = -127
        else:
            self.scaleSection1 = VerifyIntRange(scaleSection1, 0, 100)
        self.scaleSection2 = VerifyIntRange(dic_ptclScale_animationParam["scaleSection2"]["value"], 0, 100)

        self.ptclRotType = PtclRotType[dic_ptclRot["type"]["value"]]
        self.initRot = Vec3RangeFromDict(dic_ptclRot["initRot"]["value"], 0.0, MATH_PI_2)
        self.initRotRand = Vec3RangeFromDict(dic_ptclRot["initRotRand"]["value"], 0.0, MATH_PI_2)
        self.rotRegist = VerifyF32Positive(dic_ptclRot["rotRegist"]["value"])
        self.rotVel = Vec3RangeFromDict(dic_ptclRot["rotVel"]["value"], -MATH_PI, MATH_PI)
        self.rotVelRand = Vec3RangeFromDict(dic_ptclRot["rotVelRand"]["value"], 0.0, MATH_PI_2)

        self.isStopEmitInFade = VerifyBool(dic_termination["isStopEmitInFade"]["value"])
        self.alphaAddInFade = VerifyF32Positive(dic_termination["alphaAddInFade"]["value"])

        callbackID = dic_userData["callbackID"]["value"]
        if callbackID is None:
            self.userCallbackID = UserDataCallBackID.Null
        else:
            self.userCallbackID = UserDataCallBackID(callbackID)
        self.userDataBit = VerifyU16(int(dic_userData["bitfield"]["value"], 2))
        numArrayU8 = dic_userData["numArrayU8"]["value"]
        assert isinstance(numArrayU8, list) and len(numArrayU8) == 6
        self.userDataU8 = tuple(VerifyU8(v) for v in numArrayU8)
        numArrayF = dic_userData["numArrayF"]["value"]
        assert isinstance(numArrayF, list) and len(numArrayF) == UserDataParamIdx.Max
        self.userDataF = tuple(VerifyF32(v) for v in numArrayF)

        self.userShaderDefine1 = VerifyNullableStr(dic_userShader_shaderType["macroDef1"]["value"])
        self.userShaderDefine2 = VerifyNullableStr(dic_userShader_shaderType["macroDef2"]["value"])
        self.userShaderSetting = VerifyIntRange(dic_userShader["localType"]["value"], 0, 8)
        self.userShaderFlag = VerifyU32(int(dic_userShader["bitfield"]["value"], 2))
        self.userShaderSwitchFlag = VerifyU32(int(dic_userShader["switchCase"]["value"], 2))
        self.userShaderParam.fromDict(dic_userShader["param"])

        fieldRandomData = nw__eft__FieldRandomData()
        fieldRandomEnable = fieldRandomData.fromDict(dic_complex_field["random"])
        fieldMagnetData = nw__eft__FieldMagnetData()
        fieldMagnetEnable = fieldMagnetData.fromDict(dic_complex_field["magnet"])
        fieldSpinData = nw__eft__FieldSpinData()
        fieldSpinEnable = fieldSpinData.fromDict(dic_complex_field["spin"])
        fieldConvergenceData = nw__eft__FieldConvergenceData()
        fieldConvergenceEnable = fieldConvergenceData.fromDict(dic_complex_field["convergence"])
        fieldPosAddData = nw__eft__FieldPosAddData()
        fieldPosAddEnable = fieldPosAddData.fromDict(dic_complex_field["posAdd"])
        fieldCollisionData = nw__eft__FieldCollisionData()
        fieldCollisionEnable = fieldCollisionData.fromDict(dic_complex_field["collision"])
        fluctuationData = nw__eft__FluctuationData()
        fluctuationEnable, fluctuationApplyAlpha, fluctuationApplyScale = fluctuationData.fromDict(dic_complex["flux"])
        childData = nw__eft__ChildData()
        childEnable = childData.fromDict(dic_complex["child"])
        stripeData = nw__eft__StripeData()
        stripeEmitterCoord = stripeData.fromDict(dic_complex["stripe"])
        stripeEnable = (self.billboardType == BillboardType.HistoricalStripe or
                        self.billboardType == BillboardType.ConsolidatedStripe)

        if self.type == EmitterType.Complex:
            self.childData = childData if childEnable else None
            if fieldRandomEnable or fieldMagnetEnable or fieldSpinEnable or fieldConvergenceEnable or fieldPosAddEnable or fieldCollisionEnable:
                self.fieldData = nw__eft__FieldData()
                self.fieldData.randomData = fieldRandomData if fieldRandomEnable else None
                self.fieldData.magnetData = fieldMagnetData if fieldMagnetEnable else None
                self.fieldData.spinData = fieldSpinData if fieldSpinEnable else None
                self.fieldData.convergenceData = fieldConvergenceData if fieldConvergenceEnable else None
                self.fieldData.posAddData = fieldPosAddData if fieldPosAddEnable else None
                self.fieldData.collisionData = fieldCollisionData if fieldCollisionEnable else None
            else:
                self.fieldData = None
            self.fluctuationApplyAlpha = fluctuationApplyAlpha
            self.fluctuationApplyScale = fluctuationApplyScale
            self.fluctuationData = fluctuationData if fluctuationEnable else None
            self.stripeEmitterCoord = stripeEmitterCoord
            self.stripeData = stripeData if stripeEnable else None

        else:
            assert not fieldRandomEnable and \
                   not fieldMagnetEnable and \
                   not fieldSpinEnable and \
                   not fieldConvergenceEnable and \
                   not fieldPosAddEnable and \
                   not fieldCollisionEnable and \
                   not fluctuationEnable and \
                   not childEnable and \
                   not stripeEnable

        self.isPolygon = VerifyBool(dic_unused["isPolygon"]["value"])
        self.isFollowAll = VerifyBool(dic_unused["isFollowAll"]["value"])
        self.isEmitterBillboardMtx = VerifyBool(dic_unused["isEmitterBillboardMtx"]["value"])
        self.isDirectional = VerifyBool(dic_unused["isDirectional"]["value"])

    def toYAML(self, file_path: str) -> None:
        emitterYamlObj = self.toDict()
        with open(file_path, 'w', encoding='utf-8') as outf:
            yaml.dump(emitterYamlObj, outf, yaml.CSafeDumper, default_flow_style=False, sort_keys=False)

    def fromYAML(self, file_path: str) -> None:
        with open(file_path, encoding='utf-8') as inf:
            emitterYamlObj = yaml.load(inf, yaml.CSafeLoader)
        self.fromDict(emitterYamlObj)


class nw__eft__VertexShaderKey:
    structSize = struct.calcsize('>4B2?2x2I16s')
    assert structSize == 0x20

    vertexBillboardTypeVariation: BillboardType
    vertexRotationVariation: VertexRotationVariation
    userShaderSetting: int
    stripeTypeVariation: StripeType
    stripeEmitterCoord: bool
    usePrimitive: bool
    userShaderFlag: int
    userShaderSwitchFlag: int
    userShaderCompileDef: str

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            vertexBillboardTypeVariation,   # Billboard type variation
            vertexRotationVariation,        # Rotation variation
            self.userShaderSetting,         # User shader setting value
            stripeTypeVariation,            # Stripe particle type variation
            self.stripeEmitterCoord,        # Is the stripe the type that follows the emitter?
            self.usePrimitive,              # Draw using primitives?
            self.userShaderFlag,            # User shader flags
            self.userShaderSwitchFlag,      # User shader switch flags
            userShaderCompileDef            # User shader compiler define
        ) = struct.unpack_from('>4B2?2x2I16s', data, pos)

        self.vertexBillboardTypeVariation = BillboardType(vertexBillboardTypeVariation)
        self.vertexRotationVariation = VertexRotationVariation(vertexRotationVariation)
        self.stripeTypeVariation = StripeType(stripeTypeVariation)

        self.userShaderCompileDef = readString(userShaderCompileDef)

    def save(self) -> bytes:
        userShaderCompileDef = self.userShaderCompileDef.encode('shift_jis').ljust(16, b'\0')
        assert len(userShaderCompileDef) == 16

        return struct.pack(
            '>4B2?2x2I16s',
            self.vertexBillboardTypeVariation,
            self.vertexRotationVariation,
            self.userShaderSetting,
            self.stripeTypeVariation,
            self.stripeEmitterCoord,
            self.usePrimitive,
            self.userShaderFlag,
            self.userShaderSwitchFlag,
            userShaderCompileDef
        )

    def isStripe(self):
        ret = (self.vertexBillboardTypeVariation == BillboardType.HistoricalStripe or
               self.vertexBillboardTypeVariation == BillboardType.ConsolidatedStripe)
        if ret:
            assert not self.usePrimitive
        return ret

    def makeKeyFromEmitterData(self, res: nw__eft__EmitterData, userDef: Optional[str]) -> None:
        self.vertexBillboardTypeVariation = res.billboardType
        self.vertexRotationVariation      = VertexRotationVariation.Use if res.ptclRotType != PtclRotType.NoWork else VertexRotationVariation.NoUse
        self.userShaderSetting            = res.userShaderSetting
        self.userShaderFlag               = res.userShaderFlag
        self.userShaderSwitchFlag         = res.userShaderSwitchFlag
        self.stripeTypeVariation          = StripeType.Billboard
        self.stripeEmitterCoord           = False
        self.usePrimitive                 = res.meshType == MeshType.Primitive

        if userDef is not None:
            self.userShaderCompileDef = userDef
        else:
            self.userShaderCompileDef = ''

        if (res.billboardType == BillboardType.HistoricalStripe or
            res.billboardType == BillboardType.ConsolidatedStripe):
            self.stripeTypeVariation = res.stripeData.stripeType
            self.stripeEmitterCoord = res.stripeEmitterCoord

    def makeKeyFromChildData(self, res: nw__eft__ChildData, userDef: Optional[str]) -> None:
        self.vertexBillboardTypeVariation = res.childBillboardType
        self.vertexRotationVariation      = VertexRotationVariation.Use if res.childRotType != PtclRotType.NoWork else VertexRotationVariation.NoUse
        self.userShaderSetting            = res.childUserShaderSetting
        self.userShaderFlag               = res.childUserShaderFlag
        self.userShaderSwitchFlag         = res.childUserShaderSwitchFlag
        self.stripeTypeVariation          = StripeType.Max
        self.stripeEmitterCoord           = False
        self.usePrimitive                 = res.childMeshType == MeshType.Primitive

        if userDef is not None:
            self.userShaderCompileDef = userDef
        else:
            self.userShaderCompileDef = ''

    def isEqual(self, key: Self) -> bool:
        return (
            self.vertexBillboardTypeVariation == key.vertexBillboardTypeVariation and
            self.vertexRotationVariation      == key.vertexRotationVariation      and
            self.userShaderSetting            == key.userShaderSetting            and
            self.userShaderFlag               == key.userShaderFlag               and
            self.userShaderSwitchFlag         == key.userShaderSwitchFlag         and
            self.stripeTypeVariation          == key.stripeTypeVariation          and
            self.stripeEmitterCoord           == key.stripeEmitterCoord           and
            self.usePrimitive                 == key.usePrimitive                 and
            self.userShaderCompileDef         == key.userShaderCompileDef
        )

    def getCompileSetting(self) -> str:
        transformModes = (
            "VERTEX_TRANSFORM_MODE_BILLBOARD",
            "VERTEX_TRANSFORM_MODE_PLATE_XY",
            "VERTEX_TRANSFORM_MODE_PLATE_XZ",
            "VERTEX_TRANSFORM_MODE_DIRECTIONAL_Y",
            "VERTEX_TRANSFORM_MODE_DIRECTIONAL_POLYGON",
            "VERTEX_TRANSFORM_MODE_STRIPE",
            "VERTEX_TRANSFORM_MODE_COMPLEX_STRIPE",
            "VERTEX_TRANSFORM_MODE_PRIMITIVE",  # (It's not possible to use this type)
            "VERTEX_TRANSFORM_MODE_Y_BILLBOARD"
        )
        rotationModes = (
            "VERTEX_ROTATION_MODE_NO_USE",
            "VERTEX_ROTATION_MODE_USE"
        )
        stripeTypes = (
            "STRIPE_TYPE_BILLBOARD",
            "STRIPE_TYPE_EMITTER_MATRIX",
            "STRIPE_TYPE_EMITTER_UP_DOWN"
        )

        compileString = [
            "#version 330",
            "#extension GL_EXT_Cafe : enable",
            "#define TARGET_CAFE",
           f"#define VERTEX_TRANSFORM_MODE {transformModes[self.vertexBillboardTypeVariation]}",
           f"#define VERTEX_ROTATION_MODE {rotationModes[self.vertexRotationVariation]}"
        ]

        if self.userShaderSetting > 0:
            compileString.extend((f"#define USR_SETTING_{self.userShaderSetting}",
                                  f"#define USR_VERTEX_SETTING_{self.userShaderSetting}"))
        else:
            compileString.extend(("#define USR_SETTING_NONE",
                                  "#define USR_VERTEX_SETTING_NONE"))

        if self.userShaderFlag:
            for i in range(32):
                if self.userShaderFlag >> i & 1:
                    compileString.extend((f"#define USR_FLAG_{i}",
                                        f"#define USR_VERTEX_FLAG_{i}"))

        if self.userShaderSwitchFlag:
            for i in range(32):
                if self.userShaderSwitchFlag >> i & 1:
                    compileString.extend((f"#define USR_SWITCH_FLAG_{i}",
                                        f"#define USR_VERTEX_SWITCH_FLAG_{i}"))

        if self.isStripe():
            if 0 <= self.stripeTypeVariation < StripeType.Max:
                compileString.append("#define " + stripeTypes[self.stripeTypeVariation])

            if self.stripeEmitterCoord:
                compileString.append("#define STRIPE_TYPE_EMITTER_UP_DOWN")

        elif self.usePrimitive:
            compileString.append("#define MESH_TYPE_PRIMITIVE")

        if self.userShaderCompileDef:
            compileString.append("#define " + self.userShaderCompileDef)

        return '\n'.join(compileString)

    def toDict(self) -> DictGeneric:
        return {
            "billboardType": {
                "description": "Corresponds to `emitter.particle.shape.billboardType` attribute.",
                "value": self.vertexBillboardTypeVariation.name
            },
            "rotVariation": {
                "description": "Possible values:\n" \
                               "NoUse: `emitter.ptclRot.type` attribute is set to `NoWork`.\n" \
                               "Use:   `emitter.ptclRot.type` attribute is set to something else.",
                "value": self.vertexRotationVariation.name
            },
            "userShaderLocalType": {
                "description": "Corresponds to `emitter.userShader.localType` attribute.",
                "value": self.userShaderSetting
            },
            "stripeCalcType": {
                "description": "Corresponds to `emitter.complex.stripe.calcType` attribute.",
                "value": self.stripeTypeVariation.name
            },
            "stripeFollowEmitter": {
                "description": "Corresponds to `emitter.complex.stripe.followEmitter` attribute.",
                "value": self.stripeEmitterCoord
            },
            "usePrimitive": {
                "description": "Is `emitter.particle.shape.type` attribute set to `Primitive`?",
                "value": self.usePrimitive
            },
            "userShaderBitfield": {
                "description": "Corresponds to `emitter.userShader.bitfield` attribute.",
                "value": f'0b{self.userShaderFlag:032b}'
            },
            "userShaderSwitchCase": {
                "description": "Corresponds to `emitter.userShader.switchCase` attribute.",
                "value": self.userShaderSwitchFlag
            },
            "userShaderTypeMacroDef": {
                "description": "Corresponds to either `emitter.userShader.shaderType.macroDef1` or `macroDef2` attributes, or null.",
                "value": self.userShaderCompileDef if self.userShaderCompileDef else None
            }
        }

    def fromDict(self, dic: DictGeneric) -> None:
        self.vertexBillboardTypeVariation = BillboardType[dic["billboardType"]["value"]]
        self.vertexRotationVariation = VertexRotationVariation[dic["rotVariation"]["value"]]
        self.userShaderSetting = VerifyIntRange(dic["userShaderLocalType"]["value"], 0, 8)
        self.stripeTypeVariation = StripeType[dic["stripeCalcType"]["value"]]
        self.stripeEmitterCoord = VerifyBool(dic["stripeFollowEmitter"]["value"])
        self.usePrimitive = VerifyBool(dic["usePrimitive"]["value"])
        self.userShaderFlag = VerifyU32(int(dic["userShaderBitfield"]["value"], 2))
        self.userShaderSwitchFlag = VerifyIntRange(dic["userShaderSwitchCase"]["value"], 0, 31)
        self.userShaderCompileDef = VerifyNullableStr(dic["userShaderTypeMacroDef"]["value"])


class nw__eft__FragmentShaderKey:
    structSize = struct.calcsize('>6B?11BH2I16sH2x')
    assert structSize == 0x30

    shaderVariation: FragmentShaderVariation
    useSoftEdge: bool
    textureVariation: FragmentTextureVariation
    colorVariation: ColorCombinerType
    alphaCommonSource: AlphaCommonSource
    alphaBaseVariation: AlphaBaseCombinerType
    userShaderSetting: int
    usePrimitive: bool
    texture1ColorComposite: ColorAlphaBlendType
    texture1AlphaComposite: ColorAlphaBlendType
    primitiveColorComposite: ColorAlphaBlendType
    primitiveAlphaComposite: ColorAlphaBlendType
    texture0ColorOpt: ColorSource
    texture1ColorOpt: ColorSource
    primitiveColorOpt: ColorSource
    texture0AlphaOpt: AlphaSource
    texture1AlphaOpt: AlphaSource
    primitiveAlphaOpt: AlphaSource
    applyAlpha: bool
    userShaderFlag: int
    userShaderSwitchFlag: int
    userShaderCompileDef: str

    def load(self, data: ByteString, pos: int = 0) -> None:
        (
            shaderVariation,            # Shader variation
            useSoftEdge,                # Use soft edge?
            textureVariation,           # Texture variation
            colorVariation,             # Color variation
            alphaVariation,             # Alpha variation
            self.userShaderSetting,     # User shader setting value
            self.usePrimitive,          # Draw using primitives?
            texture1ColorComposite,     # Texture 1 color composite option
            texture1AlphaComposite,     # Texture 1 alpha composite option
            primitiveColorComposite,    # Primitive color composite option
            primitiveAlphaComposite,    # Primitive alpha composite option
            texture0ColorOpt,           # Texture 0 color option
            texture1ColorOpt,           # Texture 1 color option
            primitiveColorOpt,          # Primitive color option
            texture0AlphaOpt,           # Texture 0 alpha option
            texture1AlphaOpt,           # Texture 1 alpha option
            primitiveAlphaOpt,          # Primitive alpha option
            applyAlpha,                 # Is the refraction value reflected in the alpha value?
            _12,                        # IDK (1/2)
            self.userShaderFlag,        # User shader flags
            self.userShaderSwitchFlag,  # User shader switch flags
            userShaderCompileDef,       # User shader compiler define
            _2c                         # IDK (2/2)
        ) = struct.unpack_from('>6B?11BH2I16sH2x', data, pos)

        self.shaderVariation = FragmentShaderVariation(shaderVariation)

        assert useSoftEdge in (0, 1)
        self.useSoftEdge = bool(useSoftEdge)

        self.textureVariation = FragmentTextureVariation(textureVariation)

        self.colorVariation = ColorCombinerType(colorVariation)
        self.alphaBaseVariation, self.alphaCommonSource = AlphaCombinerType(alphaVariation).deconstruct()

        self.texture1ColorComposite = ColorAlphaBlendType(texture1ColorComposite)
        self.texture1AlphaComposite = ColorAlphaBlendType(texture1AlphaComposite)
        self.primitiveColorComposite = ColorAlphaBlendType(primitiveColorComposite)
        self.primitiveAlphaComposite = ColorAlphaBlendType(primitiveAlphaComposite)

        self.texture0ColorOpt = ColorSource(texture0ColorOpt)
        self.texture1ColorOpt = ColorSource(texture1ColorOpt)
        self.primitiveColorOpt = ColorSource(primitiveColorOpt)
        self.texture0AlphaOpt = AlphaSource(texture0AlphaOpt)
        self.texture1AlphaOpt = AlphaSource(texture1AlphaOpt)
        self.primitiveAlphaOpt = AlphaSource(primitiveAlphaOpt)

        assert applyAlpha in (0, 1)
        self.applyAlpha = bool(applyAlpha)

        assert _12 == 0
        assert _2c == 0

        self.userShaderCompileDef = readString(userShaderCompileDef)

    def save(self) -> bytes:
        userShaderCompileDef = self.userShaderCompileDef.encode('shift_jis').ljust(16, b'\0')
        assert len(userShaderCompileDef) == 16

        return struct.pack(
            '>6B?11BH2I16sH2x',
            self.shaderVariation,
            int(self.useSoftEdge),
            self.textureVariation,
            self.colorVariation,
            AlphaCombinerType.construct(self.alphaBaseVariation, self.alphaCommonSource),
            self.userShaderSetting,
            self.usePrimitive,
            self.texture1ColorComposite,
            self.texture1AlphaComposite,
            self.primitiveColorComposite,
            self.primitiveAlphaComposite,
            self.texture0ColorOpt,
            self.texture1ColorOpt,
            self.primitiveColorOpt,
            self.texture0AlphaOpt,
            self.texture1AlphaOpt,
            self.primitiveAlphaOpt,
            int(self.applyAlpha),
            0,
            self.userShaderFlag,
            self.userShaderSwitchFlag,
            userShaderCompileDef,
            0
        )

    def isEqual(self, key: Self) -> bool:
        return (
            self.shaderVariation         == key.shaderVariation           and
            self.useSoftEdge             == key.useSoftEdge               and
            self.textureVariation        == key.textureVariation          and
            self.colorVariation          == key.colorVariation            and
            self.alphaCommonSource       == key.alphaCommonSource         and
            self.alphaBaseVariation      == key.alphaBaseVariation        and
            self.userShaderSetting       == key.userShaderSetting         and
            self.usePrimitive            == key.usePrimitive              and
            self.texture1ColorComposite  == key.texture1ColorComposite    and
            self.texture1AlphaComposite  == key.texture1AlphaComposite    and
            self.primitiveColorComposite == key.primitiveColorComposite   and
            self.primitiveAlphaComposite == key.primitiveAlphaComposite   and
            self.texture0ColorOpt        == key.texture0ColorOpt          and
            self.texture1ColorOpt        == key.texture1ColorOpt          and
            self.primitiveColorOpt       == key.primitiveColorOpt         and
            self.texture0AlphaOpt        == key.texture0AlphaOpt          and
            self.texture1AlphaOpt        == key.texture1AlphaOpt          and
            self.primitiveAlphaOpt       == key.primitiveAlphaOpt         and
            self.applyAlpha              == key.applyAlpha                and
            self.userShaderFlag          == key.userShaderFlag            and
            self.userShaderSwitchFlag    == key.userShaderSwitchFlag      and
            self.userShaderCompileDef    == key.userShaderCompileDef
        )

    def makeKeyFromEmitterData(self, res: nw__eft__EmitterData, userDef: Optional[str]) -> None:
        self.shaderVariation          = res.shaderType
        self.useSoftEdge              = res.shaderUseSoftEdge
        self.textureVariation         = FragmentTextureVariation.First
        if res.texRes[TextureSlot.SecondTexture].width != 0 and res.texRes[TextureSlot.SecondTexture].height != 0:
            self.textureVariation     = FragmentTextureVariation.Second
        self.colorVariation           = res.colorCombinerType
        self.alphaCommonSource        = res.alphaCommonSource
        self.alphaBaseVariation       = res.alphaBaseCombinerType
        self.userShaderSetting        = res.userShaderSetting
        self.userShaderFlag           = res.userShaderFlag
        self.userShaderSwitchFlag     = res.userShaderSwitchFlag
        self.applyAlpha               = res.shaderApplyAlphaToRefract
        self.usePrimitive             = res.meshType == MeshType.Primitive

        self.texture1ColorComposite  = res.texture1ColorBlend
        self.texture1AlphaComposite  = res.texture1AlphaBlend
        self.primitiveColorComposite = res.primitiveColorBlend
        self.primitiveAlphaComposite = res.primitiveAlphaBlend

        self.texture0ColorOpt  = res.texture0ColorSource
        self.texture1ColorOpt  = res.texture1ColorSource
        self.primitiveColorOpt = res.primitiveColorSource
        self.texture0AlphaOpt  = res.texture0AlphaSource
        self.texture1AlphaOpt  = res.texture1AlphaSource
        self.primitiveAlphaOpt = res.primitiveAlphaSource

        if userDef is not None:
            self.userShaderCompileDef = userDef
        else:
            self.userShaderCompileDef = ''

    def makeKeyFromChildData(self, res: nw__eft__ChildData, userDef: Optional[str]) -> None:
        self.shaderVariation          = res.childShaderType
        self.useSoftEdge              = res.childShaderUseSoftEdge
        self.textureVariation         = FragmentTextureVariation.First
        self.colorVariation           = res.childCombinerType
        self.alphaCommonSource        = res.childAlphaCommonSource
        self.alphaBaseVariation       = res.childAlphaBaseCombinerType
        self.userShaderSetting        = res.childUserShaderSetting
        self.userShaderFlag           = res.childUserShaderFlag
        self.userShaderSwitchFlag     = res.childUserShaderSwitchFlag
        self.applyAlpha               = True
        self.usePrimitive             = res.childMeshType == MeshType.Primitive

        self.texture1ColorComposite  = ColorAlphaBlendType.Mod
        self.texture1AlphaComposite  = ColorAlphaBlendType.Mod
        self.primitiveColorComposite = res.primitiveColorBlend
        self.primitiveAlphaComposite = res.primitiveAlphaBlend

        self.texture0ColorOpt  = res.childTextureColorSource
        self.texture1ColorOpt  = ColorSource.RGB
        self.primitiveColorOpt = res.childPrimitiveColorSource
        self.texture0AlphaOpt  = res.childTextureAlphaSource
        self.texture1AlphaOpt  = AlphaSource.Pass
        self.primitiveAlphaOpt = res.childPrimitiveAlphaSource

        if userDef is not None:
            self.userShaderCompileDef = userDef
        else:
            self.userShaderCompileDef = ''

    def getCompileSetting(self) -> str:
        shaderModes = (
            "FRAGMENT_PARTICLE_SHADER_MODE_NORMAL",
            "FRAGMENT_PARTICLE_SHADER_MODE_REFRECT",
            "FRAGMENT_PARTICLE_SHADER_MODE_DISTORTION"
        )
        softEdge = (
            "FRAGMENT_SOFT_EDGE_NO_USE",
            "FRAGMENT_SOFT_EDGE_USE"
        )
        textureModes = (
            "FRAGMENT_TEXTURE_MODE_FIRST",
            "FRAGMENT_TEXTURE_MODE_SECOND"
        )
        colorModes = (
            "FRAGMENT_COLOR_MODE_COLOR",
            "FRAGMENT_COLOR_MODE_TEXTURE",
            "FRAGMENT_COLOR_MODE_TEXTURE_INTERPOLATE",
            "FRAGMENT_COLOR_MODE_NORMAL_ADD"
        )
        alphaModes = (
            "FRAGMENT_ALPHA_MODE_MOD",
            "FRAGMENT_ALPHA_MODE_SUB",
            "FRAGMENT_ALPHA_MODE_MOD_R",
            "FRAGMENT_ALPHA_MODE_SUB_R"
        )
        composite = (
            "FRAGMENT_COMPOSITE_MUL",
            "FRAGMENT_COMPOSITE_ADD",
            "FRAGMENT_COMPOSITE_SUB"
        )
        inputOpt = (
            "FRAGMENT_PARAMETER_INPUT_OPT_SRC",
            "FRAGMENT_PARAMETER_INPUT_OPT_ONE"
        )

        compileString = [
            "#version 330",
            "#extension GL_EXT_Cafe : enable",
            "#define TARGET_CAFE",
           f"#define FRAGMENT_PARTICLE_SHADER_MODE {shaderModes[self.shaderVariation]}",
           f"#define FRAGMENT_PARTICLE_SHADER_SOFT_EDGE {softEdge[self.useSoftEdge]}",
           f"#define FRAGMENT_TEXTURE_MODE {textureModes[self.textureVariation]}",
           f"#define FRAGMENT_COLOR_MODE {colorModes[self.colorVariation]}",
           f"#define FRAGMENT_ALPHA_MODE {alphaModes[AlphaCombinerType.construct(self.alphaBaseVariation, self.alphaCommonSource)]}",
           f"#define FRAGMENT_TEXTURE_COLOR_COMPOSITE {composite[self.texture1ColorComposite]}",
           f"#define FRAGMENT_TEXTURE_ALPHA_COMPOSITE {composite[self.texture1AlphaComposite]}",
           f"#define FRAGMENT_PRIMITIVE_COLOR_COMPOSITE {composite[self.primitiveColorComposite]}",
           f"#define FRAGMENT_PRIMITIVE_ALPHA_COMPOSITE {composite[self.primitiveAlphaComposite]}",
           f"#define FRAGMENT_FIRST_TEXTURE_COLOR_OPT {inputOpt[self.texture0ColorOpt]}",
           f"#define FRAGMENT_SECOND_TEXTURE_COLOR_OPT {inputOpt[self.texture1ColorOpt]}",
           f"#define FRAGMENT_PRIMITIVE_COLOR_OPT {inputOpt[self.primitiveColorOpt]}",
           f"#define FRAGMENT_FIRST_TEXTURE_ALPHA_OPT {inputOpt[self.texture0AlphaOpt]}",
           f"#define FRAGMENT_SECOND_TEXTURE_ALPHA_OPT {inputOpt[self.texture1AlphaOpt]}",
           f"#define FRAGMENT_PRIMITIVE_ALPHA_OPT {inputOpt[self.primitiveAlphaOpt]}"
        ]

        if self.userShaderSetting > 0:
            compileString.extend((f"#define USR_SETTING_{self.userShaderSetting}",
                                  f"#define USR_FRAGMENT_SETTING_{self.userShaderSetting}"))
        else:
            compileString.extend(("#define USR_SETTING_NONE",
                                  "#define USR_FRAGMENT_SETTING_NONE"))

        if self.usePrimitive:
            compileString.append("#define MESH_TYPE_PRIMITIVE")

        if self.applyAlpha:
            compileString.append("#define REFRECTION_APPLY_ALPHA")

        if self.userShaderFlag:
            for i in range(32):
                if self.userShaderFlag >> i & 1:
                    compileString.extend((f"#define USR_FLAG_{i}",
                                        f"#define USR_FRAGMENT_FLAG_{i}"))

        if self.userShaderSwitchFlag:
            for i in range(32):
                if self.userShaderSwitchFlag >> i & 1:
                    compileString.extend((f"#define USR_SWITCH_FLAG_{i}",
                                        f"#define USR_FRAGMENT_SWITCH_FLAG_{i}"))

        if self.userShaderCompileDef:
            compileString.append("#define " + self.userShaderCompileDef)

        return '\n'.join(compileString)

    def toDict(self) -> DictGeneric:
        return {
            "combinerMode": {
                "description": "Corresponds to `emitter.combiner.mode` attribute.",
                "value": self.shaderVariation.name
            },
            "softParticleEnable": {
                "description": "Corresponds to `emitter.combiner.softParticle.enable` attribute.",
                "value": self.useSoftEdge
            },
            "textureVariation": {
                "description": "Possible values are:\n" \
                               "First:  Emitter texture 1 is not required.\n" \
                               "Second: Emitter texture 1 is required.",
                "value": self.textureVariation.name
            },
            "ptclColorBlendType": {
                "description": "Corresponds to `emitter.combiner.colorCombiner.ptclColor.colorBlendType` attribute.",
                "value": self.colorVariation.name
            },
            "alphaCommonSource": {
                "description": "Corresponds to `emitter.combiner.alphaCombiner.commonSource` attribute.",
                "value": self.alphaCommonSource.name
            },
            "ptclAlphaBlendType": {
                "description": "Corresponds to `emitter.combiner.alphaCombiner.ptclAlpha.alphaBlendType` attribute.",
                "value": self.alphaBaseVariation.name
            },
            "userShaderLocalType": {
                "description": "Corresponds to `emitter.userShader.localType` attribute.",
                "value": self.userShaderSetting
            },
            "usePrimitive": {
                "description": "Is `emitter.particle.shape.type` attribute set to `Primitive`?",
                "value": self.usePrimitive
            },
            "texture1ColorBlendType": {
                "description": "Corresponds to `emitter.combiner.colorCombiner.texture1.colorBlendType` attribute.",
                "value": self.texture1ColorComposite.name
            },
            "texture1AlphaBlendType": {
                "description": "Corresponds to `emitter.combiner.alphaCombiner.texture1.alphaBlendType` attribute.",
                "value": self.texture1AlphaComposite.name
            },
            "primitiveColorBlendType": {
                "description": "Corresponds to `emitter.combiner.colorCombiner.primitive.colorBlendType` attribute.",
                "value": self.primitiveColorComposite.name
            },
            "primitiveAlphaBlendType": {
                "description": "Corresponds to `emitter.combiner.alphaCombiner.primitive.alphaBlendType` attribute.",
                "value": self.primitiveAlphaComposite.name
            },
            "texture0ColorSource": {
                "description": "Corresponds to `emitter.combiner.colorCombiner.texture0.source` attribute.",
                "value": self.texture0ColorOpt.name
            },
            "texture1ColorSource": {
                "description": "Corresponds to `emitter.combiner.colorCombiner.texture1.source` attribute.",
                "value": self.texture1ColorOpt.name
            },
            "primitiveColorSource": {
                "description": "Corresponds to `emitter.combiner.colorCombiner.primitive.source` attribute.",
                "value": self.primitiveColorOpt.name
            },
            "texture0AlphaSource": {
                "description": "Corresponds to `emitter.combiner.alphaCombiner.texture0.source` attribute.",
                "value": self.texture0AlphaOpt.name
            },
            "texture1AlphaSource": {
                "description": "Corresponds to `emitter.combiner.alphaCombiner.texture1.source` attribute.",
                "value": self.texture1AlphaOpt.name
            },
            "primitiveAlphaSource": {
                "description": "Corresponds to `emitter.combiner.alphaCombiner.primitive.source` attribute.",
                "value": self.primitiveAlphaOpt.name
            },
            "applyAlphaToRefract": {
                "description": "Corresponds to `emitter.combiner.applyAlphaToRefract` attribute.",
                "value": self.applyAlpha
            },
            "userShaderBitfield": {
                "description": "Corresponds to `emitter.userShader.bitfield` attribute.",
                "value": f'0b{self.userShaderFlag:032b}'
            },
            "userShaderSwitchCase": {
                "description": "Corresponds to `emitter.userShader.switchCase` attribute.",
                "value": self.userShaderSwitchFlag
            },
            "userShaderTypeMacroDef": {
                "description": "Corresponds to either `emitter.userShader.shaderType.macroDef1` or `macroDef2` attributes, or null.",
                "value": self.userShaderCompileDef if self.userShaderCompileDef else None
            }
        }

    def fromDict(self, dic: DictGeneric) -> None:
        self.shaderVariation = FragmentShaderVariation[dic["combinerMode"]["value"]]
        self.useSoftEdge = VerifyBool(dic["softParticleEnable"]["value"])
        self.textureVariation = FragmentTextureVariation[dic["textureVariation"]["value"]]
        self.colorVariation = ColorCombinerType[dic["ptclColorBlendType"]["value"]]
        self.alphaCommonSource = AlphaCommonSource[dic["alphaCommonSource"]["value"]]
        self.alphaBaseVariation = AlphaBaseCombinerType[dic["ptclAlphaBlendType"]["value"]]
        self.userShaderSetting = VerifyIntRange(dic["userShaderLocalType"]["value"], 0, 8)
        self.usePrimitive = VerifyBool(dic["usePrimitive"]["value"])
        self.texture1ColorComposite = ColorAlphaBlendType[dic["texture1ColorBlendType"]["value"]]
        self.texture1AlphaComposite = ColorAlphaBlendType[dic["texture1AlphaBlendType"]["value"]]
        self.primitiveColorComposite = ColorAlphaBlendType[dic["primitiveColorBlendType"]["value"]]
        self.primitiveAlphaComposite = ColorAlphaBlendType[dic["primitiveAlphaBlendType"]["value"]]
        self.texture0ColorOpt = ColorSource[dic["texture0ColorSource"]["value"]]
        self.texture1ColorOpt = ColorSource[dic["texture1ColorSource"]["value"]]
        self.primitiveColorOpt = ColorSource[dic["primitiveColorSource"]["value"]]
        self.texture0AlphaOpt = AlphaSource[dic["texture0AlphaSource"]["value"]]
        self.texture1AlphaOpt = AlphaSource[dic["texture1AlphaSource"]["value"]]
        self.primitiveAlphaOpt = AlphaSource[dic["primitiveAlphaSource"]["value"]]
        self.applyAlpha = VerifyBool(dic["applyAlphaToRefract"]["value"])
        self.userShaderFlag = VerifyU32(int(dic["userShaderBitfield"]["value"], 2))
        self.userShaderSwitchFlag = VerifyIntRange(dic["userShaderSwitchCase"]["value"], 0, 31)
        self.userShaderCompileDef = VerifyNullableStr(dic["userShaderTypeMacroDef"]["value"])


class nw__eft__GeometryShaderKey:
    structSize = struct.calcsize('>B3x')
    assert structSize == 4

    def load(self, data: ByteString, pos: int = 0) -> None:
        stripeType = struct.unpack_from('>B3x', data, pos)[0]
        assert stripeType == 0

    def save(self) -> bytes:
        return struct.pack('>B3x', 0)


class nw__eft__ShaderInformation:
    structSize = (
        nw__eft__VertexShaderKey.structSize +
        nw__eft__FragmentShaderKey.structSize +
        nw__eft__GeometryShaderKey.structSize +
        struct.calcsize('>2I')
    )
    assert structSize == 0x5C

    vertexShaderKey: nw__eft__VertexShaderKey
    fragmentShaderKey: nw__eft__FragmentShaderKey
    shaderSize: int
    offset: int

    def __init__(self) -> None:
        self.vertexShaderKey = nw__eft__VertexShaderKey()
        self.fragmentShaderKey = nw__eft__FragmentShaderKey()

    def load(self, data: ByteString, pos: int = 0) -> None:
        geometryShaderKey = nw__eft__GeometryShaderKey()

        self.vertexShaderKey.load(data, pos); pos += nw__eft__VertexShaderKey.structSize
        self.fragmentShaderKey.load(data, pos); pos += nw__eft__FragmentShaderKey.structSize
        geometryShaderKey.load(data, pos); pos += nw__eft__GeometryShaderKey.structSize
        (
            self.shaderSize,    # Shader binary size
            self.offset         # Shader binary (offset from end of all ShaderInformation entries)
        ) = struct.unpack_from('>2I', data, pos)

    def save(self) -> bytes:
        geometryShaderKey = nw__eft__GeometryShaderKey()

        return b''.join((
            self.vertexShaderKey.save(),
            self.fragmentShaderKey.save(),
            geometryShaderKey.save(),
            struct.pack(
                '>2I',
                self.shaderSize,
                self.offset
            )
        ))


class nw__eft__ParticleShader:
    vertexShaderKey: nw__eft__VertexShaderKey
    fragmentShaderKey: nw__eft__FragmentShaderKey
    gfdShaderBuffer: ByteString

    def SetVertexShaderKey(self, key: nw__eft__VertexShaderKey):
        self.vertexShaderKey = key

    def GetVertexShaderKey(self) -> nw__eft__VertexShaderKey:
        return self.vertexShaderKey

    def SetFragmentShaderKey(self, key: nw__eft__FragmentShaderKey):
        self.fragmentShaderKey = key

    def GetFragmentShaderKey(self) -> nw__eft__FragmentShaderKey:
        return self.fragmentShaderKey

    def SetupShaderResourceDirect(self, data: ByteString) -> None:
        self.gfdShaderBuffer = data
        assert len(self.gfdShaderBuffer) >= 4 and self.gfdShaderBuffer[:4] == b'Gfx2'

    def SetupShaderResource(self, data: ByteString, shaderResourcePos: int, shaderResourceSize: int) -> None:
        self.SetupShaderResourceDirect(data[shaderResourcePos:shaderResourcePos + shaderResourceSize])

    def IsStripe(self) -> bool:
        return self.vertexShaderKey.isStripe()

    def Compile(
        self,
        vshShaderDeclarationCode: str, fshShaderDeclarationCode: str,
        vshShaderCode: str, fshShaderCode: str,
        vshUserShaderCode: str, fshUserShaderCode: str
    ) -> None:
        vshCompileSetting = self.vertexShaderKey.getCompileSetting()
        fshCompileSetting = self.fragmentShaderKey.getCompileSetting()

        vshShaderStr: str = '\n'.join((vshCompileSetting, vshShaderDeclarationCode, vshUserShaderCode, vshShaderCode)) + '\n'
        fshShaderStr: str = '\n'.join((fshCompileSetting, fshShaderDeclarationCode, fshUserShaderCode, fshShaderCode)) + '\n'

        vshFname = "temp.vert"
        fshFname = "temp.frag"
        gfdFname = "temp.gsh"

        with open(vshFname, "wb") as outf:
            outf.write(vshShaderStr.encode('shift-jis'))

        with open(fshFname, "wb") as outf:
            outf.write(fshShaderStr.encode('shift-jis'))

        cmd = (GSH_COMPILE_PATH, '-v', vshFname, '-p', fshFname, '-o', gfdFname, '-no_limit_array_syms', '-nospark', '-O')
        print(">", ' '.join(cmd))
        process = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, creationflags=subprocess.CREATE_NO_WINDOW)
        if process.returncode != 0:
            print(str(process.stdout).strip())
            sys.exit(process.returncode)

        with open(gfdFname, "rb") as inf:
            inb = inf.read()

        self.SetupShaderResourceDirect(inb)

        os.remove(vshFname)
        os.remove(fshFname)
        os.remove(gfdFname)

    def toDict(self) -> DictGeneric:
        return {
            "description": "Definition of a key used to search the correspondent shader for each emitter.",
            "vertexShaderKey": self.vertexShaderKey.toDict(),
            "fragmentShaderKey": self.fragmentShaderKey.toDict()
        }

    def fromDict(self, dic: DictGeneric) -> None:
        self.vertexShaderKey = nw__eft__VertexShaderKey()
        self.vertexShaderKey.fromDict(dic["vertexShaderKey"])

        self.fragmentShaderKey = nw__eft__FragmentShaderKey()
        self.fragmentShaderKey.fromDict(dic["fragmentShaderKey"])

    def toYAML(self, file_path: str) -> None:
        shaderYamlObj = self.toDict()
        with open(file_path, 'w', encoding='utf-8') as outf:
            yaml.dump(shaderYamlObj, outf, yaml.CSafeDumper, default_flow_style=False, sort_keys=False)

    def fromYAML(self, file_path: str) -> None:
        with open(file_path, encoding='utf-8') as inf:
            shaderYamlObj = yaml.load(inf, yaml.CSafeLoader)
        self.fromDict(shaderYamlObj)


class nw__eft__ResourceEmitterSet:
    name: str
    userDataBit: int
    userDataNum1: int
    userDataNum2: int
    emitters: List[nw__eft__EmitterData]

    def toDict(self, emitters: Optional[List[str]] = None) -> DictGeneric:
        emitterNames = [emitter.name for emitter in self.emitters]
        if emitters is None:
            emitters = emitterNames
        else:
            assert len(emitters) == len(emitterNames)

        return {
            "userData": {
                "bitfield": {
                    "description": "16-bit bitfield.",
                    "value": f'0b{self.userDataBit:016b}'
                },
                "num1": {
                    "description": "1-byte number 1.",
                    "value": self.userDataNum1
                },
                "num2": {
                    "description": "1-byte number 2.",
                    "value": self.userDataNum2
                }
            },
            "emitters": [
                {
                    "name": {
                        "description": "Name of the emitter.",
                        "value": emitterName
                    },
                    "path": {
                        "description": "Emitter folder path, relative to this file.",
                        "value": emitterFolder
                    }
                } for emitterName, emitterFolder in zip(emitterNames, emitters)
            ]
        }

    def fromDict(self, dic: DictGeneric) -> List[DictGeneric]:
        dic_userData: DictGeneric = dic["userData"]

        self.userDataBit = VerifyU16(int(dic_userData["bitfield"]["value"], 2))
        self.userDataNum1 = VerifyU8(dic_userData["num1"]["value"])
        self.userDataNum2 = VerifyU8(dic_userData["num2"]["value"])

        emitters = dic["emitters"]
        assert isinstance(emitters, list) and len(emitters) > 0
        return emitters

    def toYAML(self, file_path: str, emitters: Optional[List[str]] = None) -> None:
        setYamlObj = self.toDict(emitters)
        with open(file_path, 'w', encoding='utf-8') as outf:
            yaml.dump(setYamlObj, outf, yaml.CSafeDumper, default_flow_style=False, sort_keys=False)

    def fromYAML(self, file_path: str) -> List[DictGeneric]:
        with open(file_path, encoding='utf-8') as inf:
            setYamlObj = yaml.load(inf, yaml.CSafeLoader)
        return self.fromDict(setYamlObj)


class nw__eft__Resource:
    name: Optional[str]
    emitterSets: List[nw__eft__ResourceEmitterSet]
    textureObjList: List[Union[GX2Texture, Image.Image]]
    vshParticleDeclarationShaderCode: str
    fshParticleDeclarationShaderCode: str
    vshParticleShaderCode: str
    fshParticleShaderCode: str
    vshStripeShaderCode: str
    vshUserShaderCode: str
    fshUserShaderCode: str
    shaders: List[nw__eft__ParticleShader]
    primitives: List[nw__eft__Primitive]

    def CreateFtexbTextureHandle(self, binData: ByteString, textureTblPos: int, texRes: nw__eft__TextureRes) -> None:
        assert TextureFormat.Unorm_RGBA8 <= texRes.nativeDataFormat <= TextureFormat.SRGB_RGBA8

        textureData_pos = textureTblPos + texRes.nativeDataPos
        textureData = bytes(binData[textureData_pos:textureData_pos + texRes.nativeDataSize])

        infoKey = (
            texRes.width,
            texRes.height,
            texRes.tileMode,
            texRes.swizzle,
            texRes.mipLevel,
            texRes.mipOffset[:max(texRes.mipLevel, 1) - 1],
            texRes.nativeDataFormat,
            texRes.nativeDataSize
        )
        key = texRes.nativeDataPos
        if key in self._textureKeySetTmp:
            index = self._textureKeySetTmp[key]
            refKey = self._textureInfoKeyListTmp[index]
            assert infoKey == refKey
            texRes.gx2TextureIndex = index
            return

        gx2Texture = GX2Texture.initTexture(
            GX2SurfaceDim.Dim2D,
            texRes.width,
            texRes.height,
            1,
            texRes.mipLevel,
            textureFormatTable[texRes.nativeDataFormat - TextureFormat.Unorm_RGBA8],
            texRes.compSel,
            texRes.tileMode,
            texRes.swizzle
        )
        surface = gx2Texture.surface
        surface.imageData = textureData[:surface.imageSize]
        if surface.numMips > 1:
            assert surface.mipSize > 0
            mipOffset = surface.mipOffset[0]
            dataSize = mipOffset + surface.mipSize
            surface.mipData = textureData[mipOffset:dataSize]
        else:
            assert surface.mipSize == 0
            dataSize = surface.imageSize
            surface.mipData = b''
        assert texRes.nativeDataSize == dataSize
        for i in range(13):
            # Interesting bug, Nintendo...
            if i < 3:
                assert texRes.mipOffset[i] == surface.mipOffset[i]
            else:
                assert texRes.mipOffset[i] == 0

        index = len(self._textureKeySetTmp)
        self._textureKeySetTmp[key] = index
        self._textureInfoKeyListTmp.append(infoKey)
        self.textureObjList.append(gx2Texture)

        texRes.gx2TextureIndex = index

    def CreateOriginalTextureHandle(self, binData: ByteString, textureTblPos: int, texRes: nw__eft__TextureRes) -> None:
        assert texRes.originalDataFormat in (TextureFormat.Unorm_RGB8, TextureFormat.Unorm_RGBA8)

        textureData_pos = textureTblPos + texRes.originalDataPos
        textureData = bytes(binData[textureData_pos:textureData_pos + texRes.originalDataSize])

        infoKey = (
            texRes.width,
            texRes.height,
            texRes.originalDataFormat,
            texRes.originalDataSize
        )
        key = texRes.originalDataPos
        if key in self._textureKeySetTmp:
            index = self._textureKeySetTmp[key]
            refKey = self._textureInfoKeyListTmp[index]
            assert infoKey == refKey
            texRes.imageIndex = index
            return

        mode = 'RGBA' if texRes.originalDataFormat == TextureFormat.Unorm_RGBA8 else 'RGB'
        image = Image.frombytes(mode, (texRes.width, texRes.height), textureData)

        index = len(self._textureKeySetTmp)
        self._textureKeySetTmp[key] = index
        self._textureInfoKeyListTmp.append(infoKey)
        self.textureObjList.append(image)

        texRes.imageIndex = index

    def __init__(self) -> None:
        self.name = None

        self.emitterSets = []
        self.textureObjList = []

        self.vshParticleDeclarationShaderCode = ''
        self.fshParticleDeclarationShaderCode = ''
        self.vshParticleShaderCode = ''
        self.fshParticleShaderCode = ''
        self.vshStripeShaderCode = ''
        self.vshUserShaderCode = ''
        self.fshUserShaderCode = ''
        self.shaders = []

        self.primitives = []

    def load(self, binData: ByteString) -> None:
        header = nw__eft__HeaderData()
        header.load(binData)

        assert header.numEmitterSet > 0

        nameTblPos = header.nameTblPos
        textureTblPos = header.textureTblPos

        shaderImageInfo_pos = header.shaderTblPos
        shaderImageInfo = nw__eft__ShaderImageInformation()
        shaderImageInfo.load(binData, shaderImageInfo_pos)

        assert shaderImageInfo.totalSize == header.shaderTblSize

        shaderSrcInfo_pos = shaderImageInfo_pos + nw__eft__ShaderImageInformation.structSize
        shaderSrcInfo = nw__eft__ShaderSrcInformation()
        shaderSrcInfo.load(binData, shaderSrcInfo_pos)

        shaderSrcTop_pos = shaderSrcInfo_pos + nw__eft__ShaderSrcInformation.structSize

        ### eft_ParticleDeclaration.vsh
        vshParticleDeclarationShaderCode_pos = shaderSrcTop_pos + shaderSrcInfo.vshParticleDeclaration.offset
        vshParticleDeclarationShaderCode_size = shaderSrcInfo.vshParticleDeclaration.size
        self.vshParticleDeclarationShaderCode = binData[vshParticleDeclarationShaderCode_pos:vshParticleDeclarationShaderCode_pos+vshParticleDeclarationShaderCode_size].decode('shift_jis')

        ### eft_ParticleDeclaration.fsh
        fshParticleDeclarationShaderCode_pos = shaderSrcTop_pos + shaderSrcInfo.fshParticleDeclaration.offset
        fshParticleDeclarationShaderCode_size = shaderSrcInfo.fshParticleDeclaration.size
        self.fshParticleDeclarationShaderCode = binData[fshParticleDeclarationShaderCode_pos:fshParticleDeclarationShaderCode_pos+fshParticleDeclarationShaderCode_size].decode('shift_jis')

        ### eft_Particle.vsh
        vshParticleShaderCode_pos = shaderSrcTop_pos + shaderSrcInfo.vshParticle.offset
        vshParticleShaderCode_size = shaderSrcInfo.vshParticle.size
        self.vshParticleShaderCode = binData[vshParticleShaderCode_pos:vshParticleShaderCode_pos+vshParticleShaderCode_size].decode('shift_jis')

        ### eft_Particle.fsh
        fshParticleShaderCode_pos = shaderSrcTop_pos + shaderSrcInfo.fshParticle.offset
        fshParticleShaderCode_size = shaderSrcInfo.fshParticle.size
        self.fshParticleShaderCode = binData[fshParticleShaderCode_pos:fshParticleShaderCode_pos+fshParticleShaderCode_size].decode('shift_jis')

        ### eft_Stripe.vsh
        vshStripeShaderCode_pos = shaderSrcTop_pos + shaderSrcInfo.vshStripe.offset
        vshStripeShaderCode_size = shaderSrcInfo.vshStripe.size
        self.vshStripeShaderCode = binData[vshStripeShaderCode_pos:vshStripeShaderCode_pos+vshStripeShaderCode_size].decode('shift_jis')

        ### eft_Stripe.gsh (We assume this shader is always disabled)

        ### UsrShader.vsh
        vshUserShaderCode_pos = shaderSrcTop_pos + shaderSrcInfo.vshUser.offset
        vshUserShaderCode_size = shaderSrcInfo.vshUser.size
        self.vshUserShaderCode = binData[vshUserShaderCode_pos:vshUserShaderCode_pos+vshUserShaderCode_size].decode('shift_jis')

        ### UsrShader.fsh
        fshUserShaderCode_pos = shaderSrcTop_pos + shaderSrcInfo.fshUser.offset
        fshUserShaderCode_size = shaderSrcInfo.fshUser.size
        self.fshUserShaderCode = binData[fshUserShaderCode_pos:fshUserShaderCode_pos+fshUserShaderCode_size].decode('shift_jis')

        shaderCodeSpan = [
            (vshParticleDeclarationShaderCode_pos, vshParticleDeclarationShaderCode_size),
            (fshParticleDeclarationShaderCode_pos, fshParticleDeclarationShaderCode_size),
            (vshParticleShaderCode_pos, vshParticleShaderCode_size),
            (fshParticleShaderCode_pos, fshParticleShaderCode_size),
            (vshStripeShaderCode_pos, vshStripeShaderCode_size),
            (vshUserShaderCode_pos, vshUserShaderCode_size),
            (fshUserShaderCode_pos, fshUserShaderCode_size)
        ]
        shaderCodeSpan.sort(key=lambda shaderTuple: shaderTuple[0])

        for i in reversed(range(1, len(shaderCodeSpan))):
            if shaderCodeSpan[i][0] == shaderCodeSpan[0][0]:
                del shaderCodeSpan[i]

        for i in range(len(shaderCodeSpan) - 1):
            assert shaderCodeSpan[i+1][0] == shaderCodeSpan[i][0] + shaderCodeSpan[i][1]

        assert shaderSrcInfo.shaderSourceNum == (
            bool(self.vshParticleDeclarationShaderCode) +
            bool(self.fshParticleDeclarationShaderCode) +
            bool(self.vshParticleShaderCode) +
            bool(self.fshParticleShaderCode) +
            bool(self.vshStripeShaderCode) +
            bool(self.vshUserShaderCode) +
            bool(self.fshUserShaderCode)
        )

        shaderNum = shaderImageInfo.shaderNum

        shaderInfo_pos = shaderImageInfo_pos + shaderImageInfo.offsetShaderBinInfo
        shaderBinTop_pos = shaderInfo_pos + shaderNum * nw__eft__ShaderInformation.structSize

        self.shaders = [nw__eft__ParticleShader() for _ in range(shaderNum)]
        for i, shader in enumerate(self.shaders):
            shaderInfo = nw__eft__ShaderInformation()
            shaderInfo.load(binData, shaderInfo_pos + i * nw__eft__ShaderInformation.structSize)

            shader.SetVertexShaderKey(shaderInfo.vertexShaderKey)
            shader.SetFragmentShaderKey(shaderInfo.fragmentShaderKey)
            shader.SetupShaderResource(binData, shaderBinTop_pos + shaderInfo.offset, shaderInfo.shaderSize)

        primitiveImageInfo_pos = header.primitiveTblPos
        primitiveImageInfo = nw__eft__PrimitiveImageInformation()
        primitiveImageInfo.load(binData, primitiveImageInfo_pos)

        assert primitiveImageInfo.totalSize == header.primitiveTblSize

        primitiveNum = primitiveImageInfo.primitiveNum

        primitiveInfoTop_pos = primitiveImageInfo_pos + nw__eft__PrimitiveImageInformation.structSize
        primitiveTableStart_pos = primitiveInfoTop_pos + primitiveNum * nw__eft__PrimitiveTableInfo.structSize

        self.primitives = [nw__eft__Primitive() for _ in range(primitiveNum)]
        for i, primitive in enumerate(self.primitives):
            primitiveInfo = nw__eft__PrimitiveTableInfo()
            primitiveInfo.load(binData, primitiveInfoTop_pos + i * nw__eft__PrimitiveTableInfo.structSize)

            position_pos = primitiveInfo.pos.offset      + primitiveTableStart_pos
            texCrd_pos   = primitiveInfo.texCoord.offset + primitiveTableStart_pos
            index_pos    = primitiveInfo.index.offset    + primitiveTableStart_pos

            normal_pos = -1
            if primitiveInfo.normal.offset > 0:
                normal_pos = primitiveInfo.normal.offset + primitiveTableStart_pos
            else:
                assert primitiveInfo.normal.size == 0

            color_pos = -1
            if primitiveInfo.color.offset > 0:
                color_pos = primitiveInfo.color.offset + primitiveTableStart_pos
            else:
                assert primitiveInfo.color.size == 0

            primitive.Initialize(binData,
                position_pos, primitiveInfo.pos.size,       primitiveInfo.pos.column,
                normal_pos,   primitiveInfo.normal.size,    primitiveInfo.normal.column,
                color_pos,    primitiveInfo.color.size,     primitiveInfo.color.column,
                texCrd_pos,   primitiveInfo.texCoord.size,  primitiveInfo.texCoord.column,
                index_pos,    primitiveInfo.index.size,     primitiveInfo.index.column)

        self._textureKeySetTmp = {}
        self._textureInfoKeyListTmp = []
        self.textureObjList = []

        self.name = None

        self.emitterSets = [nw__eft__ResourceEmitterSet() for _ in range(header.numEmitterSet)]
        for i, resSet in enumerate(self.emitterSets):
            setData = nw__eft__EmitterSetData()
            setData.load(binData, nw__eft__HeaderData.structSize + i * nw__eft__EmitterSetData.structSize)

            # Check if name is available
            if i == 0 and setData.namePos > header.namePos:
                self.name = readString(binData, header.namePos, allow_empty=False)

            resSet.name = readString(binData, nameTblPos + setData.namePos, allow_empty=False)

            resSet.userDataNum1 = setData.userDataNum1
            resSet.userDataNum2 = setData.userDataNum2
            resSet.userDataBit = setData.userDataBit

            assert setData.numEmitter > 0
            assert setData.emitterTblPos > 0

            resSet.emitters = [nw__eft__EmitterData() for _ in range(setData.numEmitter)]
            for j, emitter in enumerate(resSet.emitters):
                e = nw__eft__EmitterTblData()
                e.load(binData, setData.emitterTblPos + j * nw__eft__EmitterTblData.structSize)
                assert e.emitterPos > 0

                emitter.load(binData, e.emitterPos)
                emitter.name = readString(binData, nameTblPos + emitter.namePos, allow_empty=False)

                texRes0 = emitter.texRes[TextureSlot.FirstTexture]
                if texRes0.nativeDataSize > 0:
                    self.CreateFtexbTextureHandle(binData, textureTblPos, texRes0)
                if texRes0.originalDataSize > 0 and (texRes0.nativeDataSize <= 0 or texRes0.originalDataPos > 0):
                    self.CreateOriginalTextureHandle(binData, textureTblPos, texRes0)
                assert texRes0.gx2TextureIndex <  0 or texRes0.imageIndex <  0  # Don't allow both to be present
                if texRes0.gx2TextureIndex >= 0 or texRes0.imageIndex >= 0:
                    assert texRes0.width != 0 and texRes0.height != 0
                    assert texRes0.originalDataFormat in (TextureFormat.Unorm_RGB8, TextureFormat.Unorm_RGBA8)
                    assert texRes0.originalDataSize == texRes0.width * texRes0.height * (4 if texRes0.originalDataFormat == TextureFormat.Unorm_RGBA8 else 3)
                    assert emitter.texture0ColorSource == ColorSource.RGB or emitter.texture0AlphaSource == AlphaSource.Pass
                else:
                    assert texRes0.width == 0 and texRes0.height == 0
                    assert texRes0.originalDataFormat == TextureFormat.Invalid
                    assert texRes0.originalDataSize == 0
                    # The only case in which texture 0 is allowed to be missing:
                    assert emitter.texture0ColorSource == ColorSource.One and emitter.texture0AlphaSource == AlphaSource.One

                texRes1 = emitter.texRes[TextureSlot.SecondTexture]
                if texRes1.nativeDataSize > 0:
                    self.CreateFtexbTextureHandle(binData, textureTblPos, texRes1)
                if texRes1.originalDataSize > 0 and (texRes1.nativeDataSize <= 0 or texRes1.originalDataPos > 0):
                    self.CreateOriginalTextureHandle(binData, textureTblPos, texRes1)
                assert texRes1.gx2TextureIndex < 0 or texRes1.imageIndex < 0  # Don't allow both to be present
                if texRes1.gx2TextureIndex >= 0 or texRes1.imageIndex >= 0:
                    assert texRes1.width != 0 and texRes1.height != 0
                    assert texRes1.originalDataFormat in (TextureFormat.Unorm_RGB8, TextureFormat.Unorm_RGBA8)
                    assert texRes1.originalDataSize == texRes1.width * texRes1.height * (4 if texRes1.originalDataFormat == TextureFormat.Unorm_RGBA8 else 3)
                    assert emitter.texture1ColorSource == ColorSource.RGB or emitter.texture1AlphaSource == AlphaSource.Pass
                else:
                    assert texRes1.width == 0 and texRes1.height == 0
                    assert texRes1.originalDataFormat == TextureFormat.Invalid
                    assert texRes1.originalDataSize == 0
                    assert emitter.shaderType != FragmentShaderVariation.Distortion

                if emitter.type == EmitterType.Complex and emitter.childData is not None:
                    childData = emitter.childData
                    childTex = childData.childTex
                    if childTex.nativeDataSize > 0:
                        self.CreateFtexbTextureHandle(binData, textureTblPos, childTex)
                    if childTex.originalDataSize > 0 and (childTex.nativeDataSize <= 0 or childTex.originalDataPos > 0):
                        self.CreateOriginalTextureHandle(binData, textureTblPos, childTex)
                    assert childTex.gx2TextureIndex <  0 or childTex.imageIndex <  0  # Don't allow both to be present
                  # assert childTex.gx2TextureIndex >= 0 or childTex.imageIndex >= 0  # ... but at least one must be
                    if childTex.gx2TextureIndex >= 0 or childTex.imageIndex >= 0:
                        assert childTex.width != 0 and childTex.height != 0
                        assert childTex.originalDataFormat in (TextureFormat.Unorm_RGB8, TextureFormat.Unorm_RGBA8)
                        assert childTex.originalDataSize == childTex.width * childTex.height * (4 if childTex.originalDataFormat == TextureFormat.Unorm_RGBA8 else 3)
                        assert childData.childTextureColorSource == ColorSource.RGB or childData.childTextureAlphaSource == AlphaSource.Pass
                    else:
                        assert childTex.width == 0 and childTex.height == 0
                        assert childTex.originalDataFormat == TextureFormat.Invalid
                        assert childTex.originalDataSize == 0
                        # The only case in which the child texture is allowed to be missing:
                        assert childData.childTextureColorSource == ColorSource.One and childData.childTextureAlphaSource == AlphaSource.One

                if emitter.animKeyTable.dataSize:
                    animKeyTable = nw__eft__AnimKeyFrameInfoArray()
                    emitter.animKeyTable.animKeyTable = animKeyTable
                    animKeyTable.load(binData, header.animKeyTblPos + emitter.animKeyTable.animPos)

        # Make sure no shady business is happening
        assert len(self._textureKeySetTmp) == len(self._textureInfoKeyListTmp) == len(self.textureObjList)

        del self._textureKeySetTmp
        del self._textureInfoKeyListTmp

        for resSet in self.emitterSets:
            for emitter in resSet.emitters:
                self.verifyEmitterShader(emitter)

    def save(self) -> bytes:
        numEmitterSet = len(self.emitterSets)
        assert numEmitterSet > 0

        totalShaderSize = sum(len(shader.gfdShaderBuffer) for shader in self.shaders)

        header = nw__eft__HeaderData()
        header.numEmitterSet = numEmitterSet
        header.namePos = 0
        header.totalShaderSize = totalShaderSize

      # header.nameTblPos        # Set later
      # header.textureTblPos     # ^^
      # header.textureTblSize    # ^^
      # header.shaderTblPos      # ^^
      # header.shaderTblSize     # ^^
      # header.animKeyTblPos     # ^^
      # header.animKeyTblSize    # ^^
      # header.primitiveTblPos   # ^^
      # header.primitiveTblSize  # ^^
      # header.totalEmitterSize  # ^^

        shaderSrcInfo = nw__eft__ShaderSrcInformation()
        shaderSrcData = bytearray()
        shaderSourceNum = 0

        if self.vshParticleShaderCode:
            shaderSrcInfo.vshParticle.offset = len(shaderSrcData)
            encoded_str = self.vshParticleShaderCode.encode('shift_jis')
            shaderSrcInfo.vshParticle.size = len(encoded_str)
            shaderSrcData += encoded_str
            shaderSourceNum += 1
        else:
            shaderSrcInfo.vshParticle.offset = 0
            shaderSrcInfo.vshParticle.size = 0

        if self.fshParticleShaderCode:
            shaderSrcInfo.fshParticle.offset = len(shaderSrcData)
            encoded_str = self.fshParticleShaderCode.encode('shift_jis')
            shaderSrcInfo.fshParticle.size = len(encoded_str)
            shaderSrcData += encoded_str
            shaderSourceNum += 1
        else:
            shaderSrcInfo.fshParticle.offset = 0
            shaderSrcInfo.fshParticle.size = 0

        if self.vshStripeShaderCode:
            shaderSrcInfo.vshStripe.offset = len(shaderSrcData)
            encoded_str = self.vshStripeShaderCode.encode('shift_jis')
            shaderSrcInfo.vshStripe.size = len(encoded_str)
            shaderSrcData += encoded_str
            shaderSourceNum += 1
        else:
            shaderSrcInfo.vshStripe.offset = 0
            shaderSrcInfo.vshStripe.size = 0

        if self.vshUserShaderCode:
            shaderSrcInfo.vshUser.offset = len(shaderSrcData)
            encoded_str = self.vshUserShaderCode.encode('shift_jis')
            shaderSrcInfo.vshUser.size = len(encoded_str)
            shaderSrcData += encoded_str
            shaderSourceNum += 1
        else:
            shaderSrcInfo.vshUser.offset = 0
            shaderSrcInfo.vshUser.size = 0

        if self.fshUserShaderCode:
            shaderSrcInfo.fshUser.offset = len(shaderSrcData)
            encoded_str = self.fshUserShaderCode.encode('shift_jis')
            shaderSrcInfo.fshUser.size = len(encoded_str)
            shaderSrcData += encoded_str
            shaderSourceNum += 1
        else:
            shaderSrcInfo.fshUser.offset = 0
            shaderSrcInfo.fshUser.size = 0

        if self.vshParticleDeclarationShaderCode:
            shaderSrcInfo.vshParticleDeclaration.offset = len(shaderSrcData)
            encoded_str = self.vshParticleDeclarationShaderCode.encode('shift_jis')
            shaderSrcInfo.vshParticleDeclaration.size = len(encoded_str)
            shaderSrcData += encoded_str
            shaderSourceNum += 1
        else:
            shaderSrcInfo.vshParticleDeclaration.offset = 0
            shaderSrcInfo.vshParticleDeclaration.size = 0

        if self.fshParticleDeclarationShaderCode:
            shaderSrcInfo.fshParticleDeclaration.offset = len(shaderSrcData)
            encoded_str = self.fshParticleDeclarationShaderCode.encode('shift_jis')
            shaderSrcInfo.fshParticleDeclaration.size = len(encoded_str)
            shaderSrcData += encoded_str
            shaderSourceNum += 1
        else:
            shaderSrcInfo.fshParticleDeclaration.offset = 0
            shaderSrcInfo.fshParticleDeclaration.size = 0

        shaderSrcInfo.shaderSourceNum = shaderSourceNum
        shaderSrcInfo.shaderSourceTotalSize = len(shaderSrcData)

        shaderInfoData = bytearray()
        shaderDataSize = 0
        for shader in self.shaders:
            shaderInfo = nw__eft__ShaderInformation()
            shaderInfo.vertexShaderKey = shader.vertexShaderKey
            shaderInfo.fragmentShaderKey = shader.fragmentShaderKey
            shaderInfo.shaderSize = len(shader.gfdShaderBuffer)
            shaderInfo.offset = shaderDataSize
            shaderDataSize += shaderInfo.shaderSize
            shaderInfoData += shaderInfo.save()
        assert len(shaderInfoData) == len(self.shaders) * nw__eft__ShaderInformation.structSize

        offsetShaderBinInfo = nw__eft__ShaderImageInformation.structSize + nw__eft__ShaderSrcInformation.structSize + len(shaderSrcData)

        shaderImageInfo = nw__eft__ShaderImageInformation()
        shaderImageInfo.shaderNum = len(self.shaders)
        shaderImageInfo.totalSize = offsetShaderBinInfo + len(shaderInfoData) + totalShaderSize
        shaderImageInfo.offsetShaderBinInfo = offsetShaderBinInfo

        shaderTblData = b''.join((
            shaderImageInfo.save(),
            shaderSrcInfo.save(),
            shaderSrcData,
            shaderInfoData,
            *(shader.gfdShaderBuffer for shader in self.shaders)
        ))
        assert len(shaderTblData) == shaderImageInfo.totalSize

        primitiveInfoData = bytearray()
        primitiveAttrTotalData = bytearray()
        primitiveDataSize = []
        for primitive in self.primitives:
            primitiveInfo = nw__eft__PrimitiveTableInfo()
            primitiveAttrPos = len(primitiveAttrTotalData)
            primitiveAttrData = bytearray()

            posData = b''.join(struct.pack(VEC3_FMT, *vtx) for vtx in primitive.pos)
            primitiveInfo.pos.column = 3  # Vec3
            primitiveInfo.pos.offset = primitiveAttrPos + len(primitiveAttrData)
            primitiveInfo.pos.size = len(posData)
            primitiveAttrData += posData

            if primitive.nor is not None:
                norData = b''.join(struct.pack(VEC3_FMT, *vtx) for vtx in primitive.nor)
                primitiveInfo.normal.column = 3  # Vec3
                primitiveInfo.normal.offset = primitiveAttrPos + len(primitiveAttrData)
                primitiveInfo.normal.size = len(norData)
                primitiveAttrData += norData
            else:
                primitiveInfo.normal.column = 0
                primitiveInfo.normal.offset = 0
                primitiveInfo.normal.size = 0

            texData = b''.join(struct.pack(VEC2_FMT, *vtx) for vtx in primitive.tex)
            primitiveInfo.texCoord.column = 2  # Vec2
            primitiveInfo.texCoord.offset = primitiveAttrPos + len(primitiveAttrData)
            primitiveInfo.texCoord.size = len(texData)
            primitiveAttrData += texData

            if primitive.col is not None:
                colData = b''.join(struct.pack(VEC4_FMT, *vtx) for vtx in primitive.col)
                primitiveInfo.color.column = 4  # Vec4
                primitiveInfo.color.offset = primitiveAttrPos + len(primitiveAttrData)
                primitiveInfo.color.size = len(colData)
                primitiveAttrData += colData
            else:
                primitiveInfo.color.column = 0
                primitiveInfo.color.offset = 0
                primitiveInfo.color.size = 0

            idxData = b''.join(struct.pack(U32_FMT, idx) for idx in primitive.idx)
            primitiveInfo.index.column = 3  # WHAT???
            primitiveInfo.index.offset = primitiveAttrPos + len(primitiveAttrData)
            primitiveInfo.index.size = len(idxData)
            primitiveAttrData += idxData

            primitiveInfoData += primitiveInfo.save()
            primitiveAttrTotalData += primitiveAttrData
            primitiveDataSize.append([len(primitiveAttrData), False])

        assert len(primitiveInfoData) == len(self.primitives) * nw__eft__PrimitiveTableInfo.structSize

        primitiveImageInfo = nw__eft__PrimitiveImageInformation()
        primitiveImageInfo.primitiveNum = len(self.primitives)
        primitiveImageInfo.totalSize = nw__eft__PrimitiveImageInformation.structSize + len(primitiveInfoData) + len(primitiveAttrTotalData)

        primitiveTblData = b''.join((
            primitiveImageInfo.save(),
            primitiveInfoData,
            primitiveAttrTotalData
        ))
        assert len(primitiveTblData) == primitiveImageInfo.totalSize

        maxAlignment = 1
        texturePosSize = []
        textureTblData = bytearray()
        for texture in self.textureObjList:
            if isinstance(texture, GX2Texture):
                alignment = texture.surface.alignment
                maxAlignment = max(maxAlignment, alignment)
                alignedLen = align(len(textureTblData), alignment)
                padLen = alignedLen - len(textureTblData)
                textureTblData += b'\0' * padLen
                assert len(textureTblData) == alignedLen
                textureData = texture.surface.imageData
                if texture.surface.mipSize > 0:
                    textureData = bytearray(textureData)
                    mipOffset = texture.surface.mipOffset[0]
                    assert mipOffset >= texture.surface.imageSize
                    padLen = mipOffset - len(textureData)
                    textureData += b'\0' * padLen
                    assert len(textureData) == mipOffset
                    textureData += texture.surface.mipData
            else:
                assert isinstance(texture, Image.Image)
              # assert texture.mode in ('RGB', 'RGBA')
                textureData = texture.tobytes()
            texturePosSize.append((len(textureTblData), len(textureData)))
            textureTblData += textureData

        nameTblData = bytearray()
        if self.name is not None:
            assert self.name
            nameTblData += self.name.encode('shift_jis'); nameTblData.append(0)

        animKeyTblData = bytearray()
        emitterTblBasePos = nw__eft__HeaderData.structSize + numEmitterSet * nw__eft__EmitterSetData.structSize
        emitterTblTotalData = bytearray()
        totalEmitterSize = 0

        emitterSetData = bytearray()

        for resSet in self.emitterSets:
            numEmitter = len(resSet.emitters)
            emitterTblPos = emitterTblBasePos + len(emitterTblTotalData)

            namePos = len(nameTblData)
            assert resSet.name
            nameTblData += resSet.name.encode('shift_jis'); nameTblData.append(0)

            setData = nw__eft__EmitterSetData()
            setData.userDataNum1 = resSet.userDataNum1
            setData.userDataNum2 = resSet.userDataNum2
            setData.userDataBit = resSet.userDataBit
            setData.namePos = namePos
            setData.numEmitter = numEmitter
            setData.emitterTblPos = emitterTblPos
            emitterSetData += setData.save()

            emitterBasePos = emitterTblPos + numEmitter * nw__eft__EmitterTblData.structSize
            emitterTotalData = bytearray()

            emitterTblData = bytearray()

            for emitter in resSet.emitters:
                emitter.namePos = len(nameTblData)
                assert emitter.name
                nameTblData += emitter.name.encode('shift_jis'); nameTblData.append(0)

                texRes0 = emitter.texRes[TextureSlot.FirstTexture]
                assert texRes0.gx2TextureIndex < 0 or texRes0.imageIndex < 0  # Don't allow both to be present
                if texRes0.gx2TextureIndex >= 0:
                    assert texRes0.gx2TextureIndex < len(texturePosSize)
                    assert isinstance(self.textureObjList[texRes0.gx2TextureIndex], GX2Texture)
                    texRes0.nativeDataPos, texRes0.nativeDataSize = texturePosSize[texRes0.gx2TextureIndex]
                    texRes0.originalDataPos = 0
                  # texRes0.originalDataSize = 0  # Sadly, Nintendo didn't do this...
                elif texRes0.imageIndex >= 0:
                    assert texRes0.imageIndex < len(texturePosSize)
                    assert isinstance(self.textureObjList[texRes0.imageIndex], Image.Image)
                    texRes0.originalDataPos, texRes0.originalDataSize = texturePosSize[texRes0.imageIndex]
                    texRes0.nativeDataPos = 0
                    texRes0.nativeDataSize = 0
                else:
                    texRes0.nativeDataFormat = TextureFormat.Invalid
                    texRes0.nativeDataPos = 0
                    texRes0.nativeDataSize = 0
                    texRes0.originalDataFormat = TextureFormat.Invalid
                    texRes0.originalDataPos = 0
                    texRes0.originalDataSize = 0

                texRes1 = emitter.texRes[TextureSlot.SecondTexture]
                assert texRes1.gx2TextureIndex < 0 or texRes1.imageIndex < 0  # Don't allow both to be present
                if texRes1.gx2TextureIndex >= 0:
                    assert texRes1.gx2TextureIndex < len(texturePosSize)
                    assert isinstance(self.textureObjList[texRes1.gx2TextureIndex], GX2Texture)
                    texRes1.nativeDataPos, texRes1.nativeDataSize = texturePosSize[texRes1.gx2TextureIndex]
                    texRes1.originalDataPos = 0
                  # texRes1.originalDataSize = 0  # Sadly, Nintendo didn't do this...
                elif texRes1.imageIndex >= 0:
                    assert texRes1.imageIndex < len(texturePosSize)
                    assert isinstance(self.textureObjList[texRes1.imageIndex], Image.Image)
                    texRes1.originalDataPos, texRes1.originalDataSize = texturePosSize[texRes1.imageIndex]
                    texRes1.nativeDataPos = 0
                    texRes1.nativeDataSize = 0
                else:
                    texRes1.nativeDataFormat = TextureFormat.Invalid
                    texRes1.nativeDataPos = 0
                    texRes1.nativeDataSize = 0
                    texRes1.originalDataFormat = TextureFormat.Invalid
                    texRes1.originalDataPos = 0
                    texRes1.originalDataSize = 0

                if emitter.type == EmitterType.Complex and emitter.childData is not None:
                    childTex = emitter.childData.childTex
                    assert childTex.gx2TextureIndex < 0 or childTex.imageIndex < 0  # Don't allow both to be present
                    if childTex.gx2TextureIndex >= 0:
                        assert childTex.gx2TextureIndex < len(texturePosSize)
                        assert isinstance(self.textureObjList[childTex.gx2TextureIndex], GX2Texture)
                        childTex.nativeDataPos, childTex.nativeDataSize = texturePosSize[childTex.gx2TextureIndex]
                        childTex.originalDataPos = 0
                      # childTex.originalDataSize = 0  # Sadly, Nintendo didn't do this...
                    elif childTex.imageIndex >= 0:
                        assert childTex.imageIndex < len(texturePosSize)
                        assert isinstance(self.textureObjList[childTex.imageIndex], Image.Image)
                        childTex.originalDataPos, childTex.originalDataSize = texturePosSize[childTex.imageIndex]
                        childTex.nativeDataPos = 0
                        childTex.nativeDataSize = 0
                    else:
                        childTex.nativeDataFormat = TextureFormat.Invalid
                        childTex.nativeDataPos = 0
                        childTex.nativeDataSize = 0
                        childTex.originalDataFormat = TextureFormat.Invalid
                        childTex.originalDataPos = 0
                        childTex.originalDataSize = 0

                emitter.animKeyTable.animPos = len(animKeyTblData)
                if emitter.animKeyTable.animKeyTable is not None:
                    emitterAnimKeyTblData = emitter.animKeyTable.animKeyTable.save()
                    emitter.animKeyTable.dataSize = len(emitterAnimKeyTblData)
                    animKeyTblData += emitterAnimKeyTblData
                else:
                    emitter.animKeyTable.dataSize = 0

                primitiveFigure = emitter.primitiveFigure
                if primitiveFigure.index != 0xFFFFFFFF and not primitiveDataSize[primitiveFigure.index][1]:
                    primitiveFigure.dataSize = primitiveDataSize[primitiveFigure.index][0]
                    primitiveDataSize[primitiveFigure.index][1] = True
                else:
                    primitiveFigure.dataSize = 0

                if emitter.type == EmitterType.Complex and emitter.childData is not None:
                    childPrimitiveFigure = emitter.childData.childPrimitiveFigure
                    if childPrimitiveFigure.index != 0xFFFFFFFF and not primitiveDataSize[childPrimitiveFigure.index][1]:
                        childPrimitiveFigure.dataSize = primitiveDataSize[childPrimitiveFigure.index][0]
                        primitiveDataSize[childPrimitiveFigure.index][1] = True
                    else:
                        childPrimitiveFigure.dataSize = 0

                e = nw__eft__EmitterTblData()
                e.emitterPos = emitterBasePos + len(emitterTotalData)
                emitterTblData += e.save()

                emitterTotalData += emitter.save()

            emitterTblTotalData += emitterTblData
            emitterTblTotalData += emitterTotalData

            totalEmitterSize += len(emitterTotalData)

        nameTblData.append(0)  # Add an empty entry to mark the end

        # Alignments of each section:
        # Emitters section: Since it always follows HeaderData, it's already aligned
        # Texture table: Maximum alignment of contained textures
        # Name table: No alignment requirement
        # Shader table: 0x10
        # Keyframe animation table: 0x10
        # Primitive table: 0x20

        # Align texture table, which follows the emitters section
        emitterTblEndPos = emitterTblBasePos + len(emitterTblTotalData)
        emitterTblTotalData += b'\0' * (align(emitterTblEndPos, maxAlignment) - emitterTblEndPos)  # Align end to maximum texture alignment

        textureTblPos = emitterTblBasePos + len(emitterTblTotalData)
        textureTblSize = len(textureTblData)

        nameTblPos = textureTblPos + len(textureTblData)
      # nameTblSize = len(nameTblData)

        # Align shader table, which follows the name table
        nameTblEndPos = nameTblPos + len(nameTblData)
        nameTblData += b'\0' * (align(nameTblEndPos, 0x10) - nameTblEndPos)  # Align end to 0x10

        shaderTblPos = nameTblPos + len(nameTblData)
        shaderTblSize = len(shaderTblData)

        # Align keyframe animation table, which follows the shader table
        shaderTblEndPos = shaderTblPos + len(shaderTblData)
        shaderTblData += b'\0' * (align(shaderTblEndPos, 0x10) - shaderTblEndPos)  # Align end to 0x10

        animKeyTblPos = shaderTblPos + len(shaderTblData)
        animKeyTblSize = len(animKeyTblData)

        # Align primitive table, which follows the keyframe animation table
        animKeyTblEndPos = animKeyTblPos + len(animKeyTblData)
        animKeyTblData += b'\0' * (align(animKeyTblEndPos, 0x20) - animKeyTblEndPos)  # Align end to 0x20

        primitiveTblPos = animKeyTblPos + len(animKeyTblData)
        primitiveTblSize = len(primitiveTblData)

        header.nameTblPos       = nameTblPos
        header.textureTblPos    = textureTblPos
        header.textureTblSize   = textureTblSize
        header.shaderTblPos     = shaderTblPos
        header.shaderTblSize    = shaderTblSize
        header.animKeyTblPos    = animKeyTblPos
        header.animKeyTblSize   = animKeyTblSize
        header.primitiveTblPos  = primitiveTblPos
        header.primitiveTblSize = primitiveTblSize
        header.totalEmitterSize = totalEmitterSize

        return b''.join((
            header.save(),
            emitterSetData,
            emitterTblTotalData,
            textureTblData,
            nameTblData,
            shaderTblData,
            animKeyTblData,
            primitiveTblData
        ))

    def GetShader(self, vertexKey, fragmentKey) -> Optional[nw__eft__ParticleShader]:
        for shader in self.shaders:
            if shader.GetVertexShaderKey().isEqual(vertexKey) and shader.GetFragmentShaderKey().isEqual(fragmentKey):
                return shader
        return None

    @staticmethod
    def getEmitterShaderKey(res: nw__eft__EmitterData) -> List[Tuple[nw__eft__VertexShaderKey, nw__eft__FragmentShaderKey]]:
        ret: List[Tuple[nw__eft__VertexShaderKey, nw__eft__FragmentShaderKey]] = []

        for userShaderDefine in (None, res.userShaderDefine1, res.userShaderDefine2):
            if userShaderDefine is not None and not userShaderDefine:
                continue

            vertexKey = nw__eft__VertexShaderKey()
            fragmentKey = nw__eft__FragmentShaderKey()

            vertexKey  .makeKeyFromEmitterData(res, userShaderDefine)
            fragmentKey.makeKeyFromEmitterData(res, userShaderDefine)

            ret.append((vertexKey, fragmentKey))

        if res.type != EmitterType.Complex or res.childData is None:
            return ret

        cres = res.childData

        for userShaderDefine in (None, cres.childUserShaderDefine1, cres.childUserShaderDefine2):
            if userShaderDefine is not None and not userShaderDefine:
                continue

            vertexKey = nw__eft__VertexShaderKey()
            fragmentKey = nw__eft__FragmentShaderKey()

            vertexKey  .makeKeyFromChildData(cres, userShaderDefine)
            fragmentKey.makeKeyFromChildData(cres, userShaderDefine)

            ret.append((vertexKey, fragmentKey))

        return ret

    def verifyEmitterShader(self, res: nw__eft__EmitterData) -> None:
        for vertexKey, fragmentKey in self.getEmitterShaderKey(res):
            assert self.GetShader(vertexKey, fragmentKey) is not None

    def toYAML(self, file_path: str) -> None:
        assert len(self.emitterSets) > 0

        file_path = os.path.realpath(file_path)
        if os.path.isdir(file_path):
            path = file_path
            file_path = os.path.join(path, 'proj.yaml')
        else:
            path = os.path.dirname(file_path)
            if not os.path.isdir(path):
                os.mkdir(path)

        texturePath = os.path.join(path, 'texture')
        if not os.path.isdir(texturePath):
            os.mkdir(texturePath)

        for i, texture in enumerate(self.textureObjList):
            if isinstance(texture, GX2Texture):
                gfd = GFDFile()
                gfd.textures.append(texture)
                with open(os.path.join(texturePath, f'{i}.gtx'), "wb") as outf:
                    outf.write(gfd.save())
            else:
                assert isinstance(texture, Image.Image)
                texture.save(os.path.join(texturePath, f'{i}.png'), 'PNG')

        shaderBasePath = os.path.join(path, 'shader')
        if not os.path.isdir(shaderBasePath):
            os.mkdir(shaderBasePath)

        shaderSrcPath = os.path.join(shaderBasePath, 'src')
        if not os.path.isdir(shaderSrcPath):
            os.mkdir(shaderSrcPath)

        if self.vshParticleDeclarationShaderCode:
            with open(os.path.join(shaderSrcPath, 'eft_ParticleDeclaration.vsh'), 'wb') as outf:
                outf.write(self.vshParticleDeclarationShaderCode.encode('utf-8'))

        if self.fshParticleDeclarationShaderCode:
            with open(os.path.join(shaderSrcPath, 'eft_ParticleDeclaration.fsh'), 'wb') as outf:
                outf.write(self.fshParticleDeclarationShaderCode.encode('utf-8'))

        if self.vshParticleShaderCode:
            with open(os.path.join(shaderSrcPath, 'eft_Particle.vsh'), 'wb') as outf:
                outf.write(self.vshParticleShaderCode.encode('utf-8'))

        if self.fshParticleShaderCode:
            with open(os.path.join(shaderSrcPath, 'eft_Particle.fsh'), 'wb') as outf:
                outf.write(self.fshParticleShaderCode.encode('utf-8'))

        if self.vshStripeShaderCode:
            with open(os.path.join(shaderSrcPath, 'eft_Stripe.vsh'), 'wb') as outf:
                outf.write(self.vshStripeShaderCode.encode('utf-8'))

        if self.vshUserShaderCode:
            with open(os.path.join(shaderSrcPath, 'UsrShader.vsh'), 'wb') as outf:
                outf.write(self.vshUserShaderCode.encode('utf-8'))

        if self.fshUserShaderCode:
            with open(os.path.join(shaderSrcPath, 'UsrShader.fsh'), 'wb') as outf:
                outf.write(self.fshUserShaderCode.encode('utf-8'))

        shaderBinPath = os.path.join(shaderBasePath, 'bin')
        if not os.path.isdir(shaderBinPath):
            os.mkdir(shaderBinPath)

        for i, shader in enumerate(self.shaders):
            shader.toYAML(os.path.join(shaderBinPath, f'{i}_key.yaml'))

            with open(os.path.join(shaderBinPath, f'{i}.gsh'), "wb") as outf:
                outf.write(shader.gfdShaderBuffer)

        primitivePath = os.path.join(path, 'primitive')
        if not os.path.isdir(primitivePath):
            os.mkdir(primitivePath)

        for i, primitive in enumerate(self.primitives):
            primitive.toGLB(os.path.join(primitivePath, f'{i}.glb'))

        setPath = os.path.join(path, 'eset')
        if not os.path.isdir(setPath):
            os.mkdir(setPath)

        setFolderNames = []
        for resSet in self.emitterSets:
            assert len(resSet.emitters) > 0
            setFolderName = resSet.name
            i = 2
            while setFolderName in setFolderNames:
                setFolderName = resSet.name + f" ({i})"
                i += 1
            setFolderNames.append(setFolderName)
            setFolderPath = os.path.join(setPath, setFolderName)
            if not os.path.isdir(setFolderPath):
                os.mkdir(setFolderPath)

            emitterFolderNames = []
            for emitter in resSet.emitters:
                emitterFolderName = emitter.name
                i = 2
                while emitterFolderName in emitterFolderNames:
                    emitterFolderName = emitter.name + f" ({i})"
                    i += 1
                emitterFolderNames.append(emitterFolderName)
                emitterFolderPath = os.path.join(setFolderPath, emitterFolderName)
                if not os.path.isdir(emitterFolderPath):
                    os.mkdir(emitterFolderPath)

                emitter.toYAML(os.path.join(emitterFolderPath, 'data.yaml'))
                if emitter.animKeyTable.animKeyTable is not None:
                    emitter.animKeyTable.animKeyTable.toYAML(os.path.join(emitterFolderPath, 'keyframeAnimations.yaml'))

            resSet.toYAML(os.path.join(setFolderPath, 'data.yaml'), emitterFolderNames)

        proj = {
            "name": self.name,
            "esetPath": 'eset',
            "texturePath": 'texture',
            "shaderSrcPath": 'shader/src',
            "shaderBinPath": 'shader/bin',
            "primitivePath": 'primitive',
            "esets":  [
                {
                    "name": {
                        "description": "Name of the emitter set.",
                        "value": resSet.name
                    },
                    "path": {
                        "description": "Emitter set folder path, relative to `esetPath`.",
                        "value": setFolder
                    }
                } for resSet, setFolder in zip(self.emitterSets, setFolderNames)
            ]
        }
        with open(file_path, 'w', encoding='utf-8') as outf:
            yaml.dump(proj, outf, yaml.CSafeDumper, default_flow_style=False, sort_keys=False)

    def fromYAML(self, file_path: str) -> None:
        file_path = os.path.realpath(file_path)
        if not os.path.isfile(file_path):
            raise OSError("Not a file: " + file_path)
        path = os.path.dirname(file_path)
        assert os.path.isdir(path)

        with open(file_path, encoding='utf-8') as inf:
            proj = yaml.load(inf, yaml.CSafeLoader)

        self.name = proj["name"]
        assert self.name is None or isinstance(self.name, str)

        texturePath = os.path.join(path, proj["texturePath"])
        assert os.path.isdir(texturePath)

        shaderSrcPath = os.path.join(path, proj["shaderSrcPath"])
        assert os.path.isdir(shaderSrcPath)

        try:
            with open(os.path.join(shaderSrcPath, 'eft_ParticleDeclaration.vsh'), 'rb') as inf:
                self.vshParticleDeclarationShaderCode = inf.read().decode('utf-8')
        except FileNotFoundError:
            self.vshParticleDeclarationShaderCode = ''

        try:
            with open(os.path.join(shaderSrcPath, 'eft_ParticleDeclaration.fsh'), 'rb') as inf:
                self.fshParticleDeclarationShaderCode = inf.read().decode('utf-8')
        except FileNotFoundError:
            self.fshParticleDeclarationShaderCode = ''

        try:
            with open(os.path.join(shaderSrcPath, 'eft_Particle.vsh'), 'rb') as inf:
                self.vshParticleShaderCode = inf.read().decode('utf-8')
        except FileNotFoundError:
            self.vshParticleShaderCode = ''

        try:
            with open(os.path.join(shaderSrcPath, 'eft_Particle.fsh'), 'rb') as inf:
                self.fshParticleShaderCode = inf.read().decode('utf-8')
        except FileNotFoundError:
            self.fshParticleShaderCode = ''

        try:
            with open(os.path.join(shaderSrcPath, 'eft_Stripe.vsh'), 'rb') as inf:
                self.vshStripeShaderCode = inf.read().decode('utf-8')
        except FileNotFoundError:
            self.vshStripeShaderCode = ''

        try:
            with open(os.path.join(shaderSrcPath, 'UsrShader.vsh'), 'rb') as inf:
                self.vshUserShaderCode = inf.read().decode('utf-8')
        except FileNotFoundError:
            self.vshUserShaderCode = ''

        try:
            with open(os.path.join(shaderSrcPath, 'UsrShader.fsh'), 'rb') as inf:
                self.fshUserShaderCode = inf.read().decode('utf-8')
        except FileNotFoundError:
            self.fshUserShaderCode = ''

        shaderBinPath = os.path.join(path, proj["shaderBinPath"])
        assert os.path.isdir(shaderBinPath)

        primitivePath = os.path.join(path, proj["primitivePath"])
        assert os.path.isdir(primitivePath)

        setPath = os.path.join(path, proj["esetPath"])
        assert os.path.isdir(setPath)

        global LoadTextureCache, LoadPrimitiveCache
        LoadTextureCache.clear()
        LoadPrimitiveCache.clear()

        self.emitterSets = []

        esets: List[DictGeneric] = proj["esets"]
        assert isinstance(esets, list) and len(esets) > 0
        for setDic in esets:
            setName = setDic["name"]["value"]
            assert isinstance(setName, str) and setName
            setFolderName = setDic["path"]["value"]
            assert isinstance(setFolderName, str) and setFolderName
            setFolderPath = os.path.join(setPath, setFolderName)

            resSet = nw__eft__ResourceEmitterSet()
            resSet.name = setName
            resSet.emitters = []
            emitters = resSet.fromYAML(os.path.join(setFolderPath, 'data.yaml'))
            self.emitterSets.append(resSet)

            for emitterDic in emitters:
                emitterName = emitterDic["name"]["value"]
                assert isinstance(emitterName, str) and emitterName
                emitterFolderName = emitterDic["path"]["value"]
                assert isinstance(emitterFolderName, str) and emitterFolderName
                emitterFolderPath = os.path.join(setFolderPath, emitterFolderName)

                emitter = nw__eft__EmitterData()
                emitter.name = emitterName
                emitter.fromYAML(os.path.join(emitterFolderPath, 'data.yaml'))
                try:
                    animKeyTable = nw__eft__AnimKeyFrameInfoArray()
                    animKeyTable.fromYAML(os.path.join(emitterFolderPath, 'keyframeAnimations.yaml'))
                    emitter.animKeyTable.animKeyTable = animKeyTable
                except FileNotFoundError:
                    emitter.animKeyTable.animKeyTable = None
                resSet.emitters.append(emitter)

        self.textureObjList = []
        for fname in LoadTextureCache:
            if fname.endswith('.png'):
                texture = Image.open(os.path.join(texturePath, fname), formats=('PNG',))
                assert texture.mode in ('RGB', 'RGBA')
                self.textureObjList.append(texture)
            else:
                assert fname.endswith('.gtx')
                with open(os.path.join(texturePath, fname), "rb") as inf:
                    inb = inf.read()
                gfd = GFDFile()
                gfd.load(inb)
                assert len(gfd.textures) == 1
                self.textureObjList.append(gfd.textures[0])

        self.primitives = []
        for fname in LoadPrimitiveCache:
            primitive = nw__eft__Primitive()
            primitive.fromGLB(os.path.join(primitivePath, fname))
            self.primitives.append(primitive)

        shaderCache: List[nw__eft__ParticleShader] = []
        if not FORCE_RECOMPILE_SHADERS:
            for fname in os.listdir(shaderBinPath):
                if not fname.endswith('_key.yaml'):
                    continue

                shader = nw__eft__ParticleShader()
                shader.fromYAML(os.path.join(shaderBinPath, fname))

                baseFname = fname[:-9]  # len('_key.yaml') == 9
                with open(os.path.join(shaderBinPath, baseFname + '.gsh'), "rb") as inf:
                    shader.SetupShaderResourceDirect(inf.read())

                shaderCache.append(shader)

        self.shaders = []
        for resSet in self.emitterSets:
            for emitter in resSet.emitters:
                texRes0 = emitter.texRes[TextureSlot.FirstTexture]
                if texRes0.gx2TextureIndex >= 0:
                    texture = self.textureObjList[texRes0.gx2TextureIndex]
                    assert isinstance(texture, GX2Texture)
                    texRes0.fromGX2Texture(texture)
                elif texRes0.imageIndex >= 0:
                    texture = self.textureObjList[texRes0.imageIndex]
                    assert isinstance(texture, Image.Image)
                    texRes0.fromImage(texture)

                texRes1 = emitter.texRes[TextureSlot.SecondTexture]
                if texRes1.gx2TextureIndex >= 0:
                    texture = self.textureObjList[texRes1.gx2TextureIndex]
                    assert isinstance(texture, GX2Texture)
                    texRes1.fromGX2Texture(texture)
                elif texRes1.imageIndex >= 0:
                    texture = self.textureObjList[texRes1.imageIndex]
                    assert isinstance(texture, Image.Image)
                    texRes1.fromImage(texture)

                if emitter.type == EmitterType.Complex and emitter.childData is not None:
                    childTex = emitter.childData.childTex
                    if childTex.gx2TextureIndex >= 0:
                        texture = self.textureObjList[childTex.gx2TextureIndex]
                        assert isinstance(texture, GX2Texture)
                        childTex.fromGX2Texture(texture)
                    elif childTex.imageIndex >= 0:
                        texture = self.textureObjList[childTex.imageIndex]
                        assert isinstance(texture, Image.Image)
                        childTex.fromImage(texture)

                for vertexKey, fragmentKey in self.getEmitterShaderKey(emitter):
                    for shader in shaderCache:
                        if shader.GetVertexShaderKey().isEqual(vertexKey) and shader.GetFragmentShaderKey().isEqual(fragmentKey):
                            break
                    else:
                        shader = nw__eft__ParticleShader()
                        shader.SetVertexShaderKey(vertexKey)
                        shader.SetFragmentShaderKey(fragmentKey)
                        shader.Compile(
                            self.vshParticleDeclarationShaderCode, self.fshParticleDeclarationShaderCode,
                            self.vshStripeShaderCode if shader.IsStripe() else self.vshParticleShaderCode, self.fshParticleShaderCode,
                            self.vshUserShaderCode, self.fshUserShaderCode
                        )
                        shaderCache.append(shader)
                    if shader not in self.shaders:
                        self.shaders.append(shader)


def action0(ptclPath, yamlPath):
    with open(ptclPath, 'rb') as inf:
        inb = inf.read()

    res = nw__eft__Resource()
    res.load(inb)
    res.toYAML(yamlPath)


def action1(yamlPath, ptclPath):
    res = nw__eft__Resource()
    res.fromYAML(yamlPath)
    outb = res.save()

    with open(ptclPath, 'wb') as outf:
        outf.write(outb)


def main():
    if len(sys.argv) <= 1:
        action = input(
            "What would you like to do?\n" \
            "0: PTCL to YAML\n" \
            "1: YAML to PTCL\n"
        )

        if action == '0':
            ptclPath = input("Enter input PTCL file path: ")
            yamlPath = input("Enter output project YAML file path: ")
            action0(ptclPath, yamlPath)

        elif action == '1':
            yamlPath = input("Enter input project YAML file path: ")
            ptclPath = input("Enter output PTCL file path: ")
            action1(yamlPath, ptclPath)

    else:
        if len(sys.argv) != 4:
            print(
                "Expected 3 arguments!\n" \
                "To run this script with arguments, supply them in the following format:\n" \
                "PTCL to YAML: <script name> 0 <PTCL file path> <YAML file path>\n" \
                "YAML to PTCL: <script name> 1 <YAML file path> <PTCL file path>"
            )
            sys.exit(1)

        action = sys.argv[1]
        if action == '0':
            action0(sys.argv[2], sys.argv[3])

        elif action == '1':
            action1(sys.argv[2], sys.argv[3])

        else:
            print("Expected first argument to be 0 or 1!")
            sys.exit(1)


if __name__ == '__main__':
    main()
