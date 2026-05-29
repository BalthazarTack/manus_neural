#include "utils.h"
#include "auxiliary.h"

__host__ __device__ inline bool forward_mlp_siren(const float3 &g_xyz_vs,
                                                  const float3 &g_cano_xyz_vs,
                                                  const float3 &pixDir,
                                                  const float3x3 &g_inv_mat_modelview,
                                                  const float &g_inv_max_radius,
                                                  const float *viewmatrix,
                                                  const float8x3 &g_weight1,
                                                  const float8 &g_bias1,
                                                  const float8 &g_weight2,
                                                  const float g_bias2,
                                                  float &integral)
{
    integral = 0.0f;
    // Compute the intersection of the ray with the Gaussian.
    float3 dir_world = g_inv_mat_modelview * pixDir;
    float3 g_xyz_ws = g_inv_mat_modelview * g_xyz_vs;
    float A = fmaxf(EPS, dot(dir_world, dir_world));
    // float A = dot(dir_world, dir_world);
    float B = dot(g_xyz_ws, dir_world);
    float C = dot(g_xyz_ws, g_xyz_ws) - 1.0f;

    float D = B * B - A * C;

    // if (D < 1e-8f)
    if (D <= 0.0f)
        return false; // No intersection with Gaussian

    D = sqrtf(D);

    float t0 = (B - D) / A;
    float t1 = (B + D) / A;

    // MLP integration
    float3x3 inv_rot_view = make_float3x3(
        viewmatrix[0], viewmatrix[1], viewmatrix[2],
        viewmatrix[4], viewmatrix[5], viewmatrix[6],
        viewmatrix[8], viewmatrix[9], viewmatrix[10]);

    float3 d_tilde = inv_rot_view * pixDir * g_inv_max_radius;
    float3 o_tilde = inv_rot_view * g_cano_xyz_vs * (-g_inv_max_radius);
    float length_d = sqrtf(dot(d_tilde, d_tilde));
    float8 w1_tilde = g_weight1 * d_tilde;
    float8 b1_tilde = g_weight1 * o_tilde + g_bias1;

    float raw_integral = g_bias2 * (t1 - t0);

#pragma unroll
    for (int i = 0; i < 8; i++)
    {

        raw_integral += w2.data[i] / (w1_tilde.data[i] + EPS) *
                        (sinf(b1_tilde.data[i] + w1_tilde.data[i] * t1) -
                         sinf(b1_tilde.data[i] + w1_tilde.data[i] * t0));
    }

    if (raw_integral <= 0.0)
        return false; // No contribution from this Gaussian;

    integral = raw_integral * length_d;

    return true; // Successful integration
}

__host__ __device__ inline bool backward_mlp_siren(
    const float3 &g_xyz_vs,
    const float3 &g_cano_xyz_vs,
    const float3 &pixDir,
    const float3x3 &g_inv_mat_modelview,
    const float &g_inv_max_radius,
    const float *viewmatrix,
    const float8x3 &g_weight1,
    const float8 &g_bias1,
    const float8 &g_weight2,
    const float g_bias2,
    const float &integral,
    const float &dL_dintegral,
    float3 &dL_dg_xyz_vs,
    float3 &dL_dg_cano_xyz_vs,
    float3x3 &dL_dg_inv_mat_modelview,
    float &dL_dg_inv_max_radius,
    float8x3 &dL_dg_weight1,
    float8 &dL_dg_bias1,
    float8 &dL_dg_weight2,
    float &dL_dg_bias2)
{
    // Initialize gradients to zero
    dL_dg_xyz_vs = make_float3(0.0f, 0.0f, 0.0f);
    dL_dg_cano_xyz_vs = make_float3(0.0f, 0.0f, 0.0f);
    dL_dg_inv_mat_modelview = make_float3x3(0.0f);
    dL_dg_inv_max_radius = 0.0f;
    dL_dg_weight1 = make_float8x3(0.0f);
    dL_dg_bias1 = make_float8(0.0f);
    dL_dg_weight2 = make_float8(0.0f);
    dL_dg_bias2 = 0.0f;

    // Skip if forward had no contribution
    if (integral <= 0.0)
        return false;

    // Recompute forward intermediates needed for backward
    float3 dir_model = g_inv_mat_modelview * pixDir;
    float3 g_xyz_ms = g_inv_mat_modelview * g_xyz_vs;
    float A = max(EPS, dot(dir_model, dir_model));
    float B = dot(g_xyz_ms, dir_model);
    float C = dot(g_xyz_ms, g_xyz_ms) - 1;
    float D = B * B - A * C;

    if (D <= 0.0)
    {
        return false; // No intersection with Gaussian
    }

    float sqrtD = sqrt(D);
    float t0 = (B - sqrtD) / A;
    float t1 = (B + sqrtD) / A;

    float3x3 inv_rot_view = make_float3x3(
        viewmatrix[0], viewmatrix[1], viewmatrix[2],
        viewmatrix[4], viewmatrix[5], viewmatrix[6],
        viewmatrix[8], viewmatrix[9], viewmatrix[10]);

    float3 d_tilde = inv_rot_view * pixDir * g_inv_max_radius;
    float3 o_tilde = inv_rot_view * g_cano_xyz_vs * (-g_inv_max_radius);
    float length_d = sqrt(dot(d_tilde, d_tilde));
    float8 w1_tilde = g_weight1 * d_tilde;
    float8 b1_tilde = g_weight1 * o_tilde + g_bias1;

    // Compute dL_draw_integral
    float raw_integral = integral / length_d;
    float dL_draw_integral = dL_dintegral * length_d;
    float dL_dlength_d = dL_dintegral * raw_integral;

    // Compute gradients for weight2 and bias2
    dL_dg_bias2 = dL_draw_integral * (t1 - t0);

    // Compute gradients for the MLP terms
    float8 dL_dw1_tilde = make_float8(0.0);
    float8 dL_db1_tilde = make_float8(0.0);

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        float phi0_i = b1_tilde.data[i] + w1_tilde.data[i] * t0;
        float phi1_i = b1_tilde.data[i] + w1_tilde.data[i] * t1;

        float sin_phi0 = sin(phi0_i);
        float sin_phi1 = sin(phi1_i);
        float cos_phi0 = cos(phi0_i);
        float cos_phi1 = cos(phi1_i);

        float term = w1_tilde.data[i] + EPS;

        // Gradient for weight2
        dL_dg_weight2.data[i] = dL_draw_integral * (sin_phi1 - sin_phi0) / term;

        // Gradients for w1_tilde and b1_tilde
        float common = dL_draw_integral * g_weight2.data[i] / term;

        dL_dw1_tilde.data[i] = common * ((cos_phi1 * t1 - cos_phi0 * t0) -
                                         (sin_phi1 - sin_phi0) / term);

        dL_db1_tilde.data[i] = common * (cos_phi1 - cos_phi0);
    }

    // Compute gradients for t0 and t1
    float dL_dt0 = -dL_draw_integral * g_bias2;
    float dL_dt1 = dL_draw_integral * g_bias2;

#pragma unroll
    for (int i = 0; i < 8; i++)
    {
        float phi0_i = b1_tilde.data[i] + w1_tilde.data[i] * t0;
        float phi1_i = b1_tilde.data[i] + w1_tilde.data[i] * t1;

        float cos_phi0 = cos(phi0_i);
        float cos_phi1 = cos(phi1_i);

        float term = dL_draw_integral * g_weight2.data[i] / (w1_tilde.data[i] + EPS);

        dL_dt0 -= term * cos_phi0 * w1_tilde.data[i];
        dL_dt1 += term * cos_phi1 * w1_tilde.data[i];
    }

    // Compute gradients for A, B, and sqrtD
    float dL_dA = -dL_dt0 * (B - sqrtD) / (A * A) - dL_dt1 * (B + sqrtD) / (A * A);
    float dL_dB = dL_dt0 / A + dL_dt1 / A;
    float dL_dsqrtD = -dL_dt0 / A + dL_dt1 / A;

    // Compute gradients for D
    float dL_dD = dL_dsqrtD / (2.0 * sqrtD);

    // Compute gradients for A, B, C
    dL_dA += -dL_dD * C;
    dL_dB += 2.0 * dL_dD * B;
    float dL_dC = -dL_dD * A;

    // Compute gradients for dir_model and g_xyz_ms
    float3 dL_ddir_model = make_float3(0.0, 0.0, 0.0);
    float3 dL_dg_xyz_ms = make_float3(0.0, 0.0, 0.0);

    if (A > EPS)
    {
        dL_ddir_model += 2.0 * dir_model * dL_dA;
    }

    dL_ddir_model += g_xyz_ms * dL_dB;
    dL_dg_xyz_ms += dir_model * dL_dB;
    dL_dg_xyz_ms += 2.0 * g_xyz_ms * dL_dC;

    // Compute gradients for g_inv_mat_viewmodel
    dL_dg_inv_mat_modelview += outer_product(dL_ddir_model, pixDir);
    dL_dg_inv_mat_modelview += outer_product(dL_dg_xyz_ms, g_xyz_vs);

    // Compute gradients for g_xyz_vs
    dL_dg_xyz_vs += transpose(g_inv_mat_modelview) * dL_dg_xyz_ms;

    // Compute gradients for d_tilde and o_tilde
    float3 dL_dd_tilde = make_float3(0.0, 0.0, 0.0);
    float3 dL_do_tilde = make_float3(0.0, 0.0, 0.0);

    // Gradient from length_d
    if (length_d > 0.0)
    {
        dL_dd_tilde += (d_tilde / length_d) * dL_dlength_d;
    }

    // Gradients from w1_tilde and b1_tilde
    for (int i = 0; i < 8; i++)
    {
        dL_dd_tilde += g_weight1.rows[i] * dL_dw1_tilde.data[i];
        dL_do_tilde += g_weight1.rows[i] * dL_db1_tilde.data[i];

        // Gradients for weight1 and bias1
        dL_dg_weight1.rows[i] =
            dL_dw1_tilde.data[i] * d_tilde +
            dL_db1_tilde.data[i] * o_tilde;

        dL_dg_bias1.data[i] += dL_db1_tilde.data[i];
    }

    // Compute gradients for inv_max_radius
    dL_dg_inv_max_radius += dot(dL_dd_tilde, inv_rot_view * pixDir);
    dL_dg_inv_max_radius += dot(dL_do_tilde, inv_rot_view * g_cano_xyz_vs * (-1.0));

    // Compute gradients for g_cano_xyz_vs (from o_tilde)
    dL_dg_cano_xyz_vs += transpose(inv_rot_view) * (dL_do_tilde * (-g_inv_max_radius));

    return true;
}
