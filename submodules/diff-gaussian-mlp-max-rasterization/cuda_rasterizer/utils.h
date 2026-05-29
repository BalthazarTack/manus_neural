#ifndef FLOAT_STRUCTS_H
#define FLOAT_STRUCTS_H

#include <cuda_runtime.h>

#define EPS 1e-7f
#define EPS_DISCRIMINANT 1e-5f

// ======================================================== //
//                                                          //
//             float3 operations                            //
//                                                          //
// ======================================================== //

__host__ __device__ inline float dot(float3 a, float3 b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ inline float3 operator*(float3 a, float b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator*(float b, float3 a)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ inline float3 operator/(float3 a, float b)
{
    return a * (1.0f / b);
}

__host__ __device__ inline float3 operator+(float3 a, float3 b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ inline float3 operator-(float3 a, float3 b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ inline float3 &operator+=(float3 &a, const float3 &b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

__host__ __device__ inline float3 &operator-=(float3 &a, const float3 &b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    return a;
}

// ======================================================== //
//                                                          //
//              float8 operations                           //
//                                                          //
// ======================================================== //

// float8
struct float8
{
    float data[8];

    __host__ __device__ float8 operator+(const float8 &v) const
    {
        float8 result;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            result.data[i] = data[i] + v.data[i];
        return result;
    }
};

__host__ __device__ inline float8 make_float8(float v0, float v1, float v2, float v3,
                                              float v4, float v5, float v6, float v7)
{
    float8 v;
    v.data[0] = v0;
    v.data[1] = v1;
    v.data[2] = v2;
    v.data[3] = v3;
    v.data[4] = v4;
    v.data[5] = v5;
    v.data[6] = v6;
    v.data[7] = v7;
    return v;
}

__host__ __device__ inline float8 make_float8(float x)
{
    float8 v;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        v.data[i] = x;
    return v;
}

// ======================================================== //
//                                                          //
//                float8x3 operations                       //
//                                                          //
// ======================================================== //
struct float8x3
{
    float3 rows[8];

    __host__ __device__ float8 operator*(const float3 &v) const
    {
        float8 result;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            result.data[i] = dot(rows[i], v);
        return result;
    }
};

__host__ __device__ inline float8x3 make_float8x3(
    float r0x, float r0y, float r0z,
    float r1x, float r1y, float r1z,
    float r2x, float r2y, float r2z,
    float r3x, float r3y, float r3z,
    float r4x, float r4y, float r4z,
    float r5x, float r5y, float r5z,
    float r6x, float r6y, float r6z,
    float r7x, float r7y, float r7z)
{
    float8x3 m;
    m.rows[0] = make_float3(r0x, r0y, r0z);
    m.rows[1] = make_float3(r1x, r1y, r1z);
    m.rows[2] = make_float3(r2x, r2y, r2z);
    m.rows[3] = make_float3(r3x, r3y, r3z);
    m.rows[4] = make_float3(r4x, r4y, r4z);
    m.rows[5] = make_float3(r5x, r5y, r5z);
    m.rows[6] = make_float3(r6x, r6y, r6z);
    m.rows[7] = make_float3(r7x, r7y, r7z);
    return m;
}

__host__ __device__ inline float8x3 make_float8x3(float x)
{
    float8x3 m;
#pragma unroll
    for (int i = 0; i < 8; ++i)
        m.rows[i] = make_float3(x, x, x);
    return m;
}

// ======================================================== //
//                                                          //
//                float3x3 operations                       //
//                                                          //
// ======================================================== //

struct float3x3
{
    float3 rows[3];

    __host__ __device__ float3 operator*(const float3 &v) const
    {
        return make_float3(
            rows[0].x * v.x + rows[0].y * v.y + rows[0].z * v.z,
            rows[1].x * v.x + rows[1].y * v.y + rows[1].z * v.z,
            rows[2].x * v.x + rows[2].y * v.y + rows[2].z * v.z);
    }

    __host__ __device__ float3x3 operator*(const float3x3 &B) const
    {
        float3x3 result;
        for (int i = 0; i < 3; i++)
        {
            float3 row = rows[i];
            result.rows[i].x = row.x * B.rows[0].x + row.y * B.rows[1].x + row.z * B.rows[2].x;
            result.rows[i].y = row.x * B.rows[0].y + row.y * B.rows[1].y + row.z * B.rows[2].y;
            result.rows[i].z = row.x * B.rows[0].z + row.y * B.rows[1].z + row.z * B.rows[2].z;
        }
        return result;
    }
};

__host__ __device__ inline float3x3 make_float3x3(float m00, float m01, float m02,
                                                  float m10, float m11, float m12,
                                                  float m20, float m21, float m22)
{
    float3x3 m;
    m.rows[0] = make_float3(m00, m01, m02);
    m.rows[1] = make_float3(m10, m11, m12);
    m.rows[2] = make_float3(m20, m21, m22);
    return m;
}

__host__ __device__ inline float3x3 make_float3x3(float x)
{
    return make_float3x3(x, x, x,
                         x, x, x,
                         x, x, x);
}

// outer product
__host__ __device__ inline float3x3 outer_product(const float3 &a, const float3 &b)
{
    return make_float3x3(
        a.x * b.x, a.x * b.y, a.x * b.z,
        a.y * b.x, a.y * b.y, a.y * b.z,
        a.z * b.x, a.z * b.y, a.z * b.z);
}

// transpose
__host__ __device__ inline float3x3 transpose(const float3x3 &M)
{
    return make_float3x3(
        M.rows[0].x, M.rows[1].x, M.rows[2].x,
        M.rows[0].y, M.rows[1].y, M.rows[2].y,
        M.rows[0].z, M.rows[1].z, M.rows[2].z);
}

// operators for float3x3
__host__ __device__ inline float3x3 operator-(const float3x3 &a, const float3x3 &b)
{
    float3x3 result;
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        result.rows[i].x = a.rows[i].x - b.rows[i].x;
        result.rows[i].y = a.rows[i].y - b.rows[i].y;
        result.rows[i].z = a.rows[i].z - b.rows[i].z;
    }
    return result;
}

__host__ __device__ inline float3x3 operator+(const float3x3 &a, const float3x3 &b)
{
    float3x3 result;
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        result.rows[i].x = a.rows[i].x + b.rows[i].x;
        result.rows[i].y = a.rows[i].y + b.rows[i].y;
        result.rows[i].z = a.rows[i].z + b.rows[i].z;
    }
    return result;
}

__host__ __device__ inline float3x3 &operator+=(float3x3 &a, const float3x3 &b)
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        a.rows[i].x += b.rows[i].x;
        a.rows[i].y += b.rows[i].y;
        a.rows[i].z += b.rows[i].z;
    }
    return a;
}

__host__ __device__ inline float3x3 &operator-=(float3x3 &a, const float3x3 &b)
{
#pragma unroll
    for (int i = 0; i < 3; ++i)
    {
        a.rows[i].x -= b.rows[i].x;
        a.rows[i].y -= b.rows[i].y;
        a.rows[i].z -= b.rows[i].z;
    }
    return a;
}

#endif // FLOAT_STRUCTS_H
