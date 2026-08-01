//*********************************************************
//
// Copyright (c) Microsoft. All rights reserved.
// This code is licensed under the MIT License (MIT).
// THIS CODE IS PROVIDED *AS IS* WITHOUT WARRANTY OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY
// IMPLIED WARRANTIES OF FITNESS FOR A PARTICULAR
// PURPOSE, MERCHANTABILITY, OR NON-INFRINGEMENT.
//
//*********************************************************

cbuffer SceneConstantBuffer : register(b0)
{
    float4x4 worldToProjection;
    float4 padding[12];
};

Texture3D<float4> g_texture : register(t0);
Texture3D<float4> g_texture1 : register(t1);
Texture3D<float4> g_texture2 : register(t2);

SamplerState g_sampler : register(s0);

struct PSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

float4x4 tmat(float3 translate)
{
    float4x4 translateMat;

    translateMat[0] = float4(1.0, 0.0, 0.0, translate.x);
    translateMat[1] = float4(0.0, 1.0, 0.0, translate.y);
    translateMat[2] = float4(0.0, 0.0, 1.0, translate.z);
    translateMat[3] = float4(0.0, 0.0, 0.0, 1.0);
    return translateMat;
}

float4x4 rmat(float4 rotation)
{
    float4x4 rotationMat;
    float w = rotation.w;
    float x = rotation.x;
    float y = rotation.y;
    float z = rotation.z;
    //rotationMat[0] = float4(1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y), 0);
    //rotationMat[1] = float4(2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x), 0);
    //rotationMat[2] = float4(2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y), 0);
    //rotationMat[3] = float4(0, 0, 0, 1);

    rotationMat[0] = float4(1 - 2 * (y * y + z * z), 2 * (x * y + w * z), 2 * (x * z - w * y), 0);
    rotationMat[1] = float4(2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x), 0);
    rotationMat[2] = float4(2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y), 0);
    rotationMat[3] = float4(0, 0, 0, 1);
    
    return rotationMat;
}

float4x4 smat(float3 scale)
{
    float4x4 scaleMat;

    scaleMat[0] = float4(scale.x, 0.0, 0.0, 0.0);
    scaleMat[1] = float4(0.0, scale.y, 0.0, 0.0);
    scaleMat[2] = float4(0.0, 0.0, scale.z, 0.0);
    scaleMat[3] = float4(0.0, 0.0, 0.0, 1.0);
    return scaleMat;
}

PSInput VSMain(float3 position : POSITION, float4 color : COLOR)
{
    PSInput result;

    float4 translate = g_texture.SampleLevel(g_sampler, color.xyz, 0);
    float4 rotation = g_texture1.SampleLevel(g_sampler, color.xyz, 0);
    float4 scale = g_texture2.SampleLevel(g_sampler, color.xyz, 0);
    if (translate.w == 0)
        scale = float4(0, 0, 0, 1);

    float4x4 translateMat = tmat(translate.xyz);
    float4x4 rotationMat = rmat(rotation);
    float4x4 scaleMat = smat(scale.xyz);


    float4x4 trs = mul(translateMat, mul(rotationMat, scaleMat));
    float4 rpos = mul(trs, float4(position, 1.0));

    result.position = mul(worldToProjection, rpos);
    result.color = color;

    return result;
}

struct GSInput
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

struct GSOutput
{
    float4 position : SV_POSITION;
    float4 color : COLOR;
};

[maxvertexcount(3)]
void GSMain(triangle GSInput input[3], inout TriangleStream<GSOutput> output)
{
    //float4 translate = g_texture2.SampleLevel(g_sampler, float3(0.5, 0.5, 0.5), 0);
    for (int i = 0; i < 3; ++i)
    {
        GSOutput o;
        o.position = input[i].position;
        o.color = input[i].color;
        output.Append(o);
    }
    output.RestartStrip();
}

float4 PSMain(GSOutput input) : SV_TARGET
{
    return input.color;
}


