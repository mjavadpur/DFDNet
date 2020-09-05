import math
import os
import random
import sys

import cv2
import dlib
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage import io
from skimage import transform as trans

from data.image_folder import make_dataset
from models import create_model
from options.test_options import TestOptions
from util import util

sys.path.append('FaceLandmarkDetection')
import face_alignment


class FaceRestorationHelper(object):

    def __init__(self):
        self.face_detector = dlib.cnn_face_detection_model_v1(
            './packages/mmod_human_face_detector.dat')

        self.shape_predictor_5 = dlib.shape_predictor(
            './packages/shape_predictor_5_face_landmarks.dat')

        self.shape_predictor_68 = dlib.shape_predictor(
            './packages/shape_predictor_68_face_landmarks.dat')

        self.similarity_trans = trans.SimilarityTransform()
        self.face_template = np.load('./packages/FFHQ_template.npy') / 2
        self.out_size = (512, 512)
        self.upsample_factor = 2

        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_imgs = []
        self.inverse_affine_matrices = []

    def read_input_image(self, img_path):
        self.input_img = dlib.load_rgb_image(img_path)

    def detect_faces(self, img_path, upsample_num_times=1):
        # Upsamples the image upsample_num_times before running
        # the face detector
        self.read_input_image(img_path)
        self.det_faces = self.face_detector(self.input_img, upsample_num_times)
        if len(self.det_faces) == 0:
            print('No face detected. Try to increase upsample_num_times.')

    def get_face_landmarks_5(self):
        for face in self.det_faces:
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)

    def get_face_landmarks_68(self):
        for face in self.cropped_imgs:
            # face detection
            det_face = self.face_detector(face, 1)
            shape = self.shape_predictor_68(
                face, det_face[0].rect)  # (should only one face)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_68.append(landmark)

    def get_affine_matrix(self, save_cropped_path=None):
        # get affine matrix for each face
        for i, landmark in enumerate(self.all_landmarks_5):
            # get affine matrix
            self.similarity_trans.estimate(landmark, self.face_template)
            affine_matrix = self.similarity_trans.params[0:2, :]
            self.affine_matrices.append(affine_matrix)  # TODO:need copy?
            # warp and crop image
            cropped_img = cv2.warpAffine(self.input_img, affine_matrix,
                                         self.out_size)  # TODO: img shape?
            self.cropped_imgs.append(cropped_img)
            if save_cropped_path is not None:
                io.imsave(save_cropped_path, cropped_img)
            # get inverse affine matrix
            self.similarity_trans.estimate(self.face_template,
                                           landmark * self.upsample_factor)
            inverse_affine = self.similarity_trans.params[0:2, :]
            self.inverse_affine_matrices.append(
                inverse_affine)  # TODO:need copy?

    def add_restored_face(self, face):
        self.restored_faces.append(face)

    def paste_to_input_image(self, save_path):
        h, w, _ = self.input_img.shape
        h_up, w_up = h * self.upsample_factor, w * self.upsample_factor

        upsample_img = cv2.resize(self.input_img, (w_up, h_up))
        for restored_face, inverse_affine in zip(self.restored_faces,
                                                 self.inverse_affine_matrices):
            inv_restored = cv2.warpAffine(restored_face, inverse_affine,
                                          (w_up, h_up))
            mask = np.ones((*self.out_size, 3), dtype=np.float32)
            inv_mask = cv2.warpAffine(mask, inverse_affine, (w_up, h_up))

            # remove the black border
            inv_mask_erosion = cv2.erode(
                inv_mask,
                np.ones((2 * self.upsample_factor, 2 * self.upsample_factor),
                        np.uint8))
            inv_restored_remove_border = inv_mask_erosion * inv_restored
            total_face_area = np.sum(inv_mask_erosion) // 3
            # compute the fusion edge based on the area of face
            w_edge = int(total_face_area**0.5) // 20
            erosion_radius = w_edge * 2
            inv_mask_center = cv2.erode(
                inv_mask_erosion,
                np.ones((erosion_radius, erosion_radius), np.uint8))
            blur_size = w_edge * 2
            inv_soft_mask = cv2.GaussianBlur(inv_mask_center,
                                             (blur_size + 1, blur_size + 1), 0)
            upsample_img = inv_soft_mask * inv_restored_remove_border + (
                1 - inv_soft_mask) * upsample_img
        io.imsave(save_path, upsample_img.astype(np.uint8))

    def clean_all(self):
        self.all_landmarks_5 = []
        self.all_landmarks_68 = []
        self.restored_faces = []
        self.affine_matrices = []
        self.cropped_imgs = []
        self.inverse_affine_matrices = []


def get_part_location(Landmarks, imgname):

    Map_LE = list(np.hstack((range(17, 22), range(36, 42))))
    Map_RE = list(np.hstack((range(22, 27), range(42, 48))))
    Map_NO = list(range(29, 36))
    Map_MO = list(range(48, 68))
    try:
        # left eye
        Mean_LE = np.mean(Landmarks[Map_LE], 0)
        L_LE = np.max((np.max(
            np.max(Landmarks[Map_LE], 0) - np.min(Landmarks[Map_LE], 0)) / 2,
                       16))
        Location_LE = np.hstack(
            (Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        # right eye
        Mean_RE = np.mean(Landmarks[Map_RE], 0)
        L_RE = np.max((np.max(
            np.max(Landmarks[Map_RE], 0) - np.min(Landmarks[Map_RE], 0)) / 2,
                       16))
        Location_RE = np.hstack(
            (Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        # nose
        Mean_NO = np.mean(Landmarks[Map_NO], 0)
        L_NO = np.max((np.max(
            np.max(Landmarks[Map_NO], 0) - np.min(Landmarks[Map_NO], 0)) / 2,
                       16))
        Location_NO = np.hstack(
            (Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        # mouth
        Mean_MO = np.mean(Landmarks[Map_MO], 0)
        L_MO = np.max((np.max(
            np.max(Landmarks[Map_MO], 0) - np.min(Landmarks[Map_MO], 0)) / 2,
                       16))
        Location_MO = np.hstack(
            (Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except Exception:
        return 0
    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(
        Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(
            0), torch.from_numpy(Location_MO).unsqueeze(0)


def obtain_inputs(img, Landmarks, img_name):
    A = Image.fromarray(img).convert('RGB')

    Part_locations = get_part_location(Landmarks, img_name)
    if Part_locations == 0:
        return 0
    C = A
    A = A.resize((512, 512), Image.BICUBIC)
    A = transforms.ToTensor()(A)
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)  #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C)  #
    return {
        'A': A.unsqueeze(0),
        'C': C.unsqueeze(0),
        'A_paths': 'path',
        'Part_locations': Part_locations
    }


if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest'  #
    official_adaption = True
    #######################################################################
    ########################### Test Param ################################
    #######################################################################
    # opt.gpu_ids = [0] # gpu id. if use cpu, set opt.gpu_ids = []
    # TestImgPath = './TestData/TestWhole' # test image path
    # ResultsDir = './Results/TestWholeResults' #save path
    # UpScaleWhole = 4  # the upsamle scale. It should be noted that our face results are fixed to 512.
    TestImgPath = opt.test_path
    ResultsDir = opt.results_dir
    UpScaleWhole = opt.upscale_factor

    # Step 1: Crop and align faces from the whole image

    SaveCropPath = os.path.join(ResultsDir, 'Step1_CropImg')
    if not os.path.exists(SaveCropPath):
        os.makedirs(SaveCropPath)

    SaveParamPath = os.path.join(
        ResultsDir, 'Step1_AffineParam')  #save the inverse affine parameters
    if not os.path.exists(SaveParamPath):
        os.makedirs(SaveParamPath)

    SaveLandmarkPath = os.path.join(ResultsDir, 'Step2_Landmarks')
    SaveRestorePath = os.path.join(
        ResultsDir, 'Step3_RestoreCropFace')  # Only Face Results

    if len(opt.gpu_ids) > 0:
        dev = 'cuda:{}'.format(opt.gpu_ids[0])
    else:
        dev = 'cpu'
    FD = face_alignment.FaceAlignment(
        face_alignment.LandmarksType._2D, device=dev, flip_input=False)

    if not os.path.exists(SaveLandmarkPath):
        os.makedirs(SaveLandmarkPath)

    if not os.path.exists(SaveRestorePath):
        os.makedirs(SaveRestorePath)
    model = create_model(opt)
    model.setup(opt)

    SaveFinalPath = os.path.join(ResultsDir, 'Step4_FinalResults')

    if not os.path.exists(SaveFinalPath):
        os.makedirs(SaveFinalPath)

    face_helper = FaceRestorationHelper()
    ImgPaths = make_dataset(TestImgPath)
    for i, ImgPath in enumerate(ImgPaths):
        ImgName = os.path.split(ImgPath)[-1]
        print('Crop and Align {} image'.format(ImgName))
        SavePath = os.path.join(SaveCropPath, ImgName)

        print('Step 1: Crop and align faces from the whole image')
        # detect face
        face_helper.detect_faces(ImgPath, upsample_num_times=1)
        face_helper.get_face_landmarks_5()
        face_helper.get_affine_matrix(SavePath)

        print('Step 2: Face landmark detection from the cropped image')

        if official_adaption:
            cropped_imgs = [io.imread(SavePath)]
        else:
            cropped_imgs = face_helper.cropped_imgs

        face_helper.get_face_landmarks_68()

        for idx, cropped_face in enumerate(cropped_imgs):

            try:
                PredsAll = FD.get_landmarks(cropped_face)
            except Exception:
                print('Error in face detection, continue...')
                continue
            if PredsAll is None:
                print('No face, continue...')
                continue
            ins = 0
            if len(PredsAll) != 1:
                hights = []
                for l in PredsAll:
                    hights.append(l[8, 1] - l[19, 1])
                ins = hights.index(max(hights))
                # print('\t################ Warning: Detected too many face, only handle the largest one...')
                # continue
            preds = PredsAll[ins]
            AddLength = np.sqrt(
                np.sum(np.power(preds[27][0:2] - preds[33][0:2], 2)))
            SaveName = ImgName + f'{idx}' + '.txt'
            np.savetxt(
                os.path.join(SaveLandmarkPath, SaveName),
                preds[:, 0:2],
                fmt='%.3f')

        print('Step 3: Face restoration')

        landmarks = []
        if not os.path.exists(
                os.path.join(SaveLandmarkPath, ImgName + '.txt')):
            print(os.path.join(SaveLandmarkPath, ImgName + '.txt'))
            print('\t################ No landmark file')
        with open(os.path.join(SaveLandmarkPath, ImgName + '.txt'), 'r') as f:
            for line in f:
                tmp = [np.float(i) for i in line.split(' ') if i != '\n']
                landmarks.append(tmp)
        landmarks = np.array(landmarks)

        for cropped_face, landmarks in zip(cropped_imgs,
                                           face_helper.all_landmarks_68):
            torch.cuda.empty_cache()
            data = obtain_inputs(cropped_face, landmarks, ImgName)
            if data == 0:
                print('Error in landmark file, continue...')
                continue
            model.set_input(data)
            try:
                model.test()
                visuals = model.get_current_visuals()
                im_data = visuals['fake_A']
                im = util.tensor2im(im_data)
                util.save_image(im, os.path.join(SaveRestorePath, ImgName))
                face_helper.add_restored_face(im)
            except Exception as e:
                print(f'Error in enhancing this image: {str(e)}. continue...')
                continue

        print('Step 4: Paste the Restored Face to the Input Image')

        for restored_face in cropped_imgs:
            WholeInputPath = os.path.join(TestImgPath, ImgName)
            FaceResultPath = os.path.join(SaveRestorePath, ImgName)
            ParamPath = os.path.join(SaveParamPath, ImgName + '.npy')
            SaveWholePath = os.path.join(SaveFinalPath, ImgName)
            face_helper.paste_to_input_image(SaveWholePath)

        face_helper.clean_all()
    print('\nAll results are saved in {} \n'.format(ResultsDir))
