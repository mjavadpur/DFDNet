import os

import cv2
import dlib
import mmcv
import numpy as np
import torch
import torchvision.transforms as transforms
from skimage import io
from skimage import transform as trans

from data.image_folder import make_dataset
from models import create_model
from options.test_options import TestOptions
from util import util


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
        return len(self.det_faces)

    def get_face_landmarks_5(self):
        for face in self.det_faces:
            print('5', face.rect)
            shape = self.shape_predictor_5(self.input_img, face.rect)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_5.append(landmark)
        return len(self.all_landmarks_5)

    def get_face_landmarks_68(self):
        for face in self.cropped_imgs:
            # face detection
            det_face = self.face_detector(face, 1)
            print('68', det_face[0].rect)
            shape = self.shape_predictor_68(
                face, det_face[0].rect)  # (should only one face)
            landmark = np.array([[part.x, part.y] for part in shape.parts()])
            self.all_landmarks_68.append(landmark)

    def warp_crop_faces(self, save_cropped_path=None):
        for idx, landmark in enumerate(self.all_landmarks_5):
            # get affine matrix
            self.similarity_trans.estimate(landmark, self.face_template)
            affine_matrix = self.similarity_trans.params[0:2, :]
            self.affine_matrices.append(affine_matrix)
            # warp and crop image
            cropped_img = cv2.warpAffine(self.input_img, affine_matrix,
                                         self.out_size)
            self.cropped_imgs.append(cropped_img)
            if save_cropped_path is not None:
                path, ext = os.path.splitext(save_cropped_path)
                save_path = f'{path}_{idx}{ext}'
                io.imsave(save_path, cropped_img)
            # get inverse affine matrix
            self.similarity_trans.estimate(self.face_template,
                                           landmark * self.upsample_factor)
            inverse_affine = self.similarity_trans.params[0:2, :]
            self.inverse_affine_matrices.append(inverse_affine)

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
    # A = Image.fromarray(img).convert('RGB')
    # A = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    A = img
    Part_locations = get_part_location(Landmarks, img_name)
    if Part_locations == 0:
        return 0
    C = A
    # A = A.resize((512, 512), Image.BICUBIC)
    A = cv2.resize(A, (512, 512), interpolation=cv2.INTER_CUBIC)
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
    official_adaption = False

    TestImgPath = opt.test_path
    ImgPaths = make_dataset(TestImgPath)
    result_root = opt.results_dir
    UpScaleWhole = opt.upscale_factor
    if len(opt.gpu_ids) > 0:
        dev = 'cuda:{}'.format(opt.gpu_ids[0])
    else:
        dev = 'cpu'
    model = create_model(opt)
    model.setup(opt)

    # Step 1: Crop and align faces from the whole image

    save_crop_root = os.path.join(result_root, 'cropped_faces')
    save_restore_root = os.path.join(result_root, 'restored_faces')
    save_final_root = os.path.join(result_root, 'final_results')
    mmcv.mkdir_or_exist(save_crop_root)
    mmcv.mkdir_or_exist(save_restore_root)
    mmcv.mkdir_or_exist(save_restore_root)

    face_helper = FaceRestorationHelper()

    for img_path in ImgPaths:
        img_name = os.path.basename(img_path)
        print(f'Processing {img_name} image')
        torch.cuda.empty_cache()

        save_crop_path = os.path.join(save_crop_root, img_name)

        # detect faces
        num_det_faces = face_helper.detect_faces(
            img_path, upsample_num_times=1)
        # get face landmarks for each face
        num_landmarks = face_helper.get_face_landmarks_5()
        print(f'\tStep 1: Detect {num_det_faces} faces, '
              f'{num_landmarks} landmarks.')
        # warp and crop each face
        face_helper.warp_crop_faces(save_crop_path)

        if official_adaption:
            cropped_imgs = [io.imread(save_crop_path)]  # TODO
        else:
            cropped_imgs = face_helper.cropped_imgs
        # get 68 landmarks
        face_helper.get_face_landmarks_68()

        print('\tStep 3: Face restoration')

        for idx, (cropped_face, landmarks) in enumerate(
                zip(cropped_imgs, face_helper.all_landmarks_68)):
            torch.cuda.empty_cache()
            data = obtain_inputs(cropped_face, landmarks, img_name)
            if data == 0:
                print('Error in landmark file, continue...')
                continue
            model.set_input(data)
            try:
                model.test()
                visuals = model.get_current_visuals()
                im_data = visuals['fake_A']
                im = util.tensor2im(im_data)
                path, ext = os.path.splitext(
                    os.path.join(save_restore_path, img_name))
                save_path = f'{path}_{idx}{ext}'
                util.save_image(im, save_path)
                face_helper.add_restored_face(im)
            except Exception as e:
                print(f'Error in enhancing this image: {str(e)}. continue...')
                continue

        print('\tStep 4: Paste the Restored Face to the Input Image')

        for restored_face in cropped_imgs:
            SaveWholePath = os.path.join(save_final_path, img_name)
            face_helper.paste_to_input_image(SaveWholePath)

        face_helper.clean_all()
    print('\nAll results are saved in {} \n'.format(result_root))
