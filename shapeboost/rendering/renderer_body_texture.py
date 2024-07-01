import random

import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat

from shapeboost.beta_decompose.beta_process2 import part_names, part_seg, joints_name_24, part_names_ids
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    OrthographicCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    HardPhongShader,
    TexturesUV,
    TexturesVertex,
    BlendParams)


def preprocess_densepose_UV(uv_path, batch_size):
    DP_UV = loadmat(uv_path)
    faces_bodyparts = torch.Tensor(DP_UV['All_FaceIndices']).squeeze()  # (13774,) face to DensePose body part mapping
    faces_densepose = torch.from_numpy((DP_UV['All_Faces'] - 1).astype(np.int64))  # (13774, 3) face to vertices indices mapping
    verts_map = torch.from_numpy(DP_UV['All_vertices'][0].astype(np.int64)) - 1  # (7829,) DensePose vertex to SMPL vertex mapping
    u_norm = torch.Tensor(DP_UV['All_U_norm'])  # (7829, 1)  # Normalised U coordinates for each vertex
    v_norm = torch.Tensor(DP_UV['All_V_norm'])  # (7829, 1)  # Normalised V coordinates for each vertex

    # RGB texture images/maps are processed into a 6 x 4 grid (atlas) of 24 textures.
    # Atlas is ordered by DensePose body parts (down rows then across columns).
    # UV coordinates for vertices need to be offset to match the texture image grid.
    offset_per_part = {}
    already_offset = set()
    cols, rows = 4, 6
    for i, u in enumerate(np.linspace(0, 1, cols, endpoint=False)):
        for j, v in enumerate(np.linspace(0, 1, rows, endpoint=False)):
            part = rows * i + j + 1  # parts are 1-indexed in face_indices
            offset_per_part[part] = (u, v)
    u_norm_offset = u_norm.clone()
    v_norm_offset = v_norm.clone()
    vertex_parts = torch.zeros(u_norm.shape[0])  # Also want to get a mapping between vertices and their corresponding DP body parts (technically one-to-many but ignoring that here).
    for i in range(len(faces_densepose)):
        face_vert_idxs = faces_densepose[i]
        part = faces_bodyparts[i]
        offset_u, offset_v = offset_per_part[int(part.item())]
        for vert_idx in face_vert_idxs:
            # vertices are reused (at DensePose part boundaries), but we don't want to offset multiple times
            if vert_idx.item() not in already_offset:
                # offset u value
                u_norm_offset[vert_idx] = u_norm_offset[vert_idx] / cols + offset_u
                # offset v value
                # this also flips each part locally, as each part is upside down
                v_norm_offset[vert_idx] = (1 - v_norm_offset[vert_idx]) / rows + offset_v
                # add vertex to our set tracking offsetted vertices
                already_offset.add(vert_idx.item())
        vertex_parts[face_vert_idxs] = part

    # invert V values
    v_norm = 1 - v_norm
    v_norm_offset = 1 - v_norm_offset

    # Combine body part indices (I), and UV coordinates
    verts_uv_offset = torch.cat([u_norm_offset[None], v_norm_offset[None]], dim=2).expand(batch_size, -1, -1)  # (batch_size, 7829, 2)
    verts_iuv = torch.cat([vertex_parts[None, :, None], u_norm[None], v_norm[None]], dim=2).expand(batch_size, -1, -1)  # (batch_size, 7829, 3)

    # Add a batch dimension to faces
    faces_densepose = faces_densepose[None].expand(batch_size, -1, -1)

    return verts_uv_offset, verts_iuv, verts_map, faces_densepose


class TextureBodyRenderer(nn.Module):
    def __init__(self,
                 device,
                 batch_size,
                 smpl_faces,
                 img_wh=256,
                 cam_t=None,
                 cam_R=None,
                 projection_type='perspective',
                 perspective_focal_length=300,
                 blur_radius=0.0,
                 faces_per_pixel=1,
                 light_t=((0.0, 0.0, -2.0),),
                 light_ambient_color=((0.5, 0.5, 0.5),),
                 light_diffuse_color=((0.3, 0.3, 0.3),),
                 light_specular_color=((0.2, 0.2, 0.2),),
                 background_color=(0.0, 0.0, 0.0)):
        global part_seg
        super().__init__()
        self.img_wh = img_wh

        # UV pre-processing for textures
        verts_uv_offset, verts_iuv, verts_map, faces_densepose = preprocess_densepose_UV(uv_path='model_files/UV_Processed.mat', batch_size=batch_size)
        self.verts_uv_offset = verts_uv_offset.to(device)
        self.verts_iuv = verts_iuv.to(device)
        self.verts_map = verts_map.to(device)
        self.faces_densepose = faces_densepose.to(device)

        # Cameras - pre-defined here but can be specified in forward pass if cameras will vary (e.g. random cameras)
        assert projection_type in ['perspective', 'orthographic'], print('Invalid projection type:', projection_type)
        print('\nRenderer projection type:', projection_type)
        self.projection_type = projection_type
        if cam_R is None:
            # Rotating 180° about z-axis to make pytorch3d camera convention same as what I've been using so far in my perspective_project_torch/NMR/pyrender
            # (Actually pyrender also has a rotation defined in the renderer to make it same as NMR.)
            cam_R = torch.tensor([[-1., 0., 0.],
                                  [0., -1., 0.],
                                  [0., 0., 1.]], device=device).float()
            cam_R = cam_R[None, :, :].expand(batch_size, -1, -1)
        if cam_t is None:
            cam_t = torch.tensor([0., 0.2, 2.5]).float().to(device)[None, :].expand(batch_size, -1)
        # Pytorch3D camera is rotated 180° about z-axis to match my perspective_project_torch/NMR's projection convention.
        # So, need to also rotate the given camera translation (implemented below as elementwise-mul).
        cam_t = cam_t * torch.tensor([-1., -1., 1.], device=cam_t.device).float()
        if projection_type == 'perspective':
            self.cameras = PerspectiveCameras(
                focal_length=((2 * perspective_focal_length / img_wh,
                            2 * perspective_focal_length / img_wh),),
                device=device,
            )

            # self.cameras_body = PerspectiveCameras(
            #     focal_length=((2 * perspective_focal_length / 64,
            #                 2 * perspective_focal_length / 64),),
            #     device=device,
            # )
            self.cameras_body = self.cameras

            self.register_buffer('cam_t', cam_t)
            self.register_buffer('cam_R', cam_R)
        elif projection_type == 'orthographic':
            raise NotImplementedError

        # Lights for textured RGB render - pre-defined here but can be specified in forward pass if lights will vary (e.g. random cameras)
        self.lights_rgb_render = PointLights(device=device,
                                                location=light_t,
                                                ambient_color=light_ambient_color,
                                                diffuse_color=light_diffuse_color,
                                                specular_color=light_specular_color)
                                            
        # Lights for IUV render - don't want lighting to affect the rendered image.
        self.lights_iuv_render = PointLights(device=device,
                                             ambient_color=[[1, 1, 1]],
                                             diffuse_color=[[0, 0, 0]],
                                             specular_color=[[0, 0, 0]])

        # Rasterizer
        raster_settings = RasterizationSettings(image_size=img_wh,
                                                blur_radius=blur_radius,
                                                faces_per_pixel=faces_per_pixel,)
        
        raster_settings_body = RasterizationSettings(image_size=64,
                                                blur_radius=blur_radius,
                                                faces_per_pixel=faces_per_pixel,
                                                bin_size=4)
                                            
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings)  # Specify camera in forward pass

        self.rasterizer_body = MeshRasterizer(cameras=self.cameras, raster_settings=raster_settings_body)

        # Shader for textured RGB output and IUV output
        blend_params = BlendParams(background_color=background_color)
        self.iuv_shader = HardPhongShader(device=device, cameras=self.cameras,
                                          lights=self.lights_iuv_render, blend_params=blend_params)

        self.rgb_shader = HardPhongShader(device=device, cameras=self.cameras,
                                            lights=self.lights_rgb_render, blend_params=blend_params)

        local_random = random.Random(0)
        self.faces_smpl = smpl_faces.unsqueeze(0).expand(batch_size, -1, -1)
        self.verts_iuv_whole = torch.ones(batch_size, 6890, 3, device=device)
        self.iuv_record = []
        for i, name in enumerate(joints_name_24):
            self.iuv_record.append( (i+1, local_random.randint(2, 255), local_random.randint(2, 255)) )
            now_part_seg = part_seg[name]
            self.verts_iuv_whole[:, now_part_seg, 0] = self.iuv_record[-1][0]
            self.verts_iuv_whole[:, now_part_seg, 1] = self.iuv_record[-1][1]
            self.verts_iuv_whole[:, now_part_seg, 2] = self.iuv_record[-1][2]

        print('self.iuv_record', self.iuv_record)

        self.iuv_record = torch.from_numpy(np.array(self.iuv_record)).to(device).type(torch.int32)
        self.to(device)
    
    def parse_smpl_imgs(self, smpl_imgs):
        # print('smpl_imgs', smpl_imgs.shape)
        original_seg1 = (smpl_imgs.float() + 0.1).type(torch.int32)
        original_seg2 = (smpl_imgs.float() - 0.1).type(torch.int32) + 1

        batch_size, height, width = smpl_imgs.shape[0], smpl_imgs.shape[1], smpl_imgs.shape[2]
        # print('batch_size, height, width',batch_size, height, width)
        x_part = torch.zeros(batch_size, 20, height, width, device=smpl_imgs.device, dtype=torch.float32)
        iuv_record = self.iuv_record
        # for i in range(24):
        #     part_seg1 = (original_seg1 == iuv_record[i]).all(dim=-1)
        #     part_seg2 = (original_seg2 == iuv_record[i]).all(dim=-1)
        #     part_seg_both = (part_seg1 & part_seg2)
        #     x_part[:, i] = part_seg_both.type(torch.int32)
        for i, name_id in enumerate(part_names_ids):
            part_seg1 = (original_seg1 == iuv_record[name_id]).all(dim=-1)
            part_seg2 = (original_seg2 == iuv_record[name_id]).all(dim=-1)
            part_seg_both = (part_seg1 & part_seg2)
            x_part[:, i] = part_seg_both.float()

        return x_part
    
    def to(self, device):
        # Rasterizer and shader have submodules which are not of type nn.Module
        self.rasterizer.to(device)
        self.rgb_shader.to(device)
        self.iuv_shader.to(device)

    def forward(self, vertices, textures=None, cam_t=None, lights_rgb_settings=None,
                verts_features=None, img_center=None):
        if img_center is not None:
            assert NotImplementedError
        else:
            cameras = self.cameras
        
        if cam_t is not None:
            vertices = vertices + cam_t[:, None, :]
        
        vertices = vertices * torch.tensor([-1., -1., 1.], device=cam_t.device).float()

        if lights_rgb_settings is not None:
            self.lights_rgb_render.location = lights_rgb_settings['location']
            self.lights_rgb_render.ambient_color = lights_rgb_settings['ambient_color']
            self.lights_rgb_render.diffuse_color = lights_rgb_settings['diffuse_color']
            self.lights_rgb_render.specular_color = lights_rgb_settings['specular_color']

        vertices_densepose = vertices[:, self.verts_map, :]  # From SMPL verts indexing (0 to 6889) to DP verts indexing (0 to 7828), verts shape is (B, 7829, 3)

        textures_iuv = TexturesVertex(verts_features=self.verts_iuv)
        meshes_iuv = Meshes(verts=vertices_densepose, faces=self.faces_densepose, textures=textures_iuv)

        if verts_features is not None:
            verts_features = verts_features[:, self.verts_map, :]  # From SMPL verts indexing (0 to 6889) to DP verts indexing (0 to 7828), verts shape is (B, 7829, 3)
            textures_rgb = TexturesVertex(verts_features=verts_features)
        else:
            textures_rgb = TexturesUV(maps=textures, faces_uvs=self.faces_densepose, verts_uvs=self.verts_uv_offset)
        meshes_rgb = Meshes(verts=vertices_densepose, faces=self.faces_densepose, textures=textures_rgb)

        # Rasterize
        fragments = self.rasterizer(meshes_iuv, cameras=cameras)
        zbuffers = fragments.zbuf[:, :, :, 0]

        # Render RGB and IUV outputs
        output = {}
        output['iuv_images'] = self.iuv_shader(fragments, meshes_iuv, lights=self.lights_iuv_render)[:, :, :, :3]

        rgb_images = self.rgb_shader(fragments, meshes_rgb, lights=self.lights_rgb_render)[:, :, :, :3]
        output['rgb_images'] = torch.clamp(rgb_images, max=1.0)

        # Get depth image
        output['depth_images'] = zbuffers

        ############### BODY MASK
        textures_iuv_smpl = TexturesVertex(verts_features=self.verts_iuv_whole)
        meshes_iuv = Meshes(verts=vertices, faces=self.faces_smpl, textures=textures_iuv_smpl)
    
        # Rasterize
        fragments = self.rasterizer_body(meshes_iuv, cameras=self.cameras_body)
        # Render RGB and IUV outputs
        smpl_part_mask = self.iuv_shader(fragments, meshes_iuv, lights=self.lights_iuv_render, cameras=self.cameras_body)[:, :, :, :3]
        smpl_part_mask = self.parse_smpl_imgs(smpl_part_mask)
        output['smpl_part_mask'] = smpl_part_mask

        return output