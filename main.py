import cv2
import argparse
import numpy as np
import logging
from ai_models.utils.image_process import pad_image_to_cube


class SCRFD:
    __logger = logging.getLogger(__qualname__)

    # output_key = ["out0", "out1", "out2", "out3", "out4", "out5", "out6", "out7", "out8"]
    output_key = ["score_8", "score_16", "score_32", "bbox_8", "bbox_16", "bbox_32", "kps_8", "kps_16", "kps_32"]

    def __init__(self, onnxmodel, confThreshold=0.5, nmsThreshold=0.5):
        self.inpWidth = 640
        self.inpHeight = 640
        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.net = cv2.dnn.readNet(onnxmodel)
        self.keep_ratio = True
        self.fmc = 3
        self._feat_stride_fpn = [8, 16, 32]
        self._num_anchors = 2

        self.anchor_centers = self._build_anchors(
            self.inpWidth, self.inpHeight, self._feat_stride_fpn, self._num_anchors
        )

    def resize_image(self, srcimg):
        padh, padw, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padw = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(
                    img, 0, 0, padw, self.inpWidth - neww - padw, cv2.BORDER_CONSTANT, value=0
                )  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale) + 1, self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                padh = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, padh, self.inpHeight - newh - padh, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, padh, padw

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = x1.clamp(min=0, max=max_shape[1])
            y1 = y1.clamp(min=0, max=max_shape[0])
            x2 = x2.clamp(min=0, max=max_shape[1])
            y2 = y2.clamp(min=0, max=max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)

    def distance2kps(self, points, distance, max_shape=None):
        preds = []
        for i in range(0, distance.shape[1], 2):
            px = points[:, i % 2] + distance[:, i]
            py = points[:, i % 2 + 1] + distance[:, i + 1]
            if max_shape is not None:
                px = px.clamp(min=0, max=max_shape[1])
                py = py.clamp(min=0, max=max_shape[0])
            preds.append(px)
            preds.append(py)
        return np.stack(preds, axis=-1)

    @staticmethod
    def _build_anchors(input_height, input_width, strides, num_anchors):
        """
        Precompute anchor points for provided image size

        :param input_height: Input image height
        :param input_width: Input image width
        :param strides: Model strides
        :param num_anchors: Model num anchors
        :return: box centers
        """

        centers = []
        for stride in strides:
            height = input_height // stride
            width = input_width // stride

            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            if num_anchors > 1:
                anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            centers.append(anchor_centers)
        return centers

    def pre_process(self, list_srcimg):
        # img, newh, neww, padh, padw = self.resize_image(srcimg)
        # scale_h, scale_w = srcimg.shape[0] / newh, srcimg.shape[1] / neww
        # blob = cv2.dnn.blobFromImage(
        #     img, 1.0 / 128, (self.inpWidth, self.inpHeight), (127.5, 127.5, 127.5), swapRB=True
        # )
        list_input_image = []
        list_scale = []
        for srcimg in list_srcimg:
            original_image, input_image, scale = pad_image_to_cube(
                srcimg,
                np.array([self.inpHeight, self.inpWidth]),
                normalize_factor=128,
                normalize_offset=-127.7,
                swapRB=True,
                single_scale=False,
                expand_dims=False,
            )
            list_input_image.append(input_image)
            list_scale.append(scale)
        input_images = np.array(list_input_image)
        scales = np.array(list_scale)
        return input_images, scales

    def detect(self, list_srcimg):
        # list_nms_bboxes = []
        # list_nms_kpss = []
        # list_nms_scores = []
        # input_image, scale = self.pre_process(srcimg)
        input_images, scales = self.pre_process(list_srcimg)

        # Sets the input to the network
        self.net.setInput(input_images)
        # self.__logger.debug(f"{pad= }, {scale= }")
        self.__logger.debug(f"{scales= }")

        # Runs the forward pass to get output of the output layers
        # outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        self.__logger.debug(f"{self.net.getUnconnectedOutLayersNames()= }")
        outs = self.net.forward(self.output_key)
        # self.__logger.debug(f"{outs.shape= }")

        self.__logger.debug(f"{len(outs)= }")
        for i, out in enumerate(outs):
            self.__logger.debug(f"outs[{i}].shape={out.shape}")
            # self.__logger.debug(f"outs[{i}]={out}")

        list_nms_bboxes, list_nms_kpss, list_nms_scores = self.postprocess(outs, scales)
        # list_nms_bboxes.append(nms_bboxes)
        # list_nms_kpss.append(nms_kpss)
        # list_nms_scores.append(nms_scores)
        return list_nms_bboxes, list_nms_kpss, list_nms_scores

    def postprocess(self, outs, scales):
        list_nms_bboxes = []
        list_nms_kpss = []
        list_nms_scores = []
        # inference output
        for batch, scale in enumerate(scales):
            scores_list, bboxes_list, kpss_list = [], [], []
            for idx, stride in enumerate(self._feat_stride_fpn):
                self.__logger.debug(f"{idx= }, {stride= }")
                scores = outs[idx][batch]
                bbox_preds = outs[idx + self.fmc][batch] * stride
                kps_preds = outs[idx + self.fmc * 2][batch] * stride

                self.__logger.debug(f"{scores.shape= }, {bbox_preds.shape= }, {kps_preds.shape= }")

                anchor_centers = self.anchor_centers[idx]
                self.__logger.debug(f"{anchor_centers.shape= }")

                # self.__logger.debug(f"{anchor_centers= }, {bbox_preds= }, {stride= }")
                bboxes = self.distance2bbox(anchor_centers, bbox_preds)
                self.__logger.debug(f"{bboxes.shape= }")

                pos_inds = np.where(scores >= self.confThreshold)[0]
                pos_scores = scores[pos_inds]
                pos_bboxes = bboxes[pos_inds]
                scores_list.append(pos_scores)
                bboxes_list.append(pos_bboxes)

                kpss = self.distance2kps(anchor_centers, kps_preds)
                # kpss = kps_preds
                kpss = kpss.reshape((kpss.shape[0], -1, 2))
                pos_kpss = kpss[pos_inds]
                kpss_list.append(pos_kpss)

            scores = np.vstack(scores_list).ravel()
            bboxes = np.vstack(bboxes_list)
            kpss = np.vstack(kpss_list)

            bboxes[:, :2] *= scale
            bboxes[:, 2:4] *= scale
            kpss[:, :, :2] *= scale

            indices = cv2.dnn.NMSBoxes(bboxes.tolist(), scores.tolist(), self.confThreshold, self.nmsThreshold)
            self.__logger.debug(f"{indices= }")
            nms_bboxes = [bboxes[i, :] for i in indices]
            nms_kpss = [kpss[i, :] for i in indices]
            nms_scores = [scores[i] for i in indices]
            list_nms_bboxes.append(nms_bboxes)
            list_nms_kpss.append(nms_kpss)
            list_nms_scores.append(nms_scores)
        return list_nms_bboxes, list_nms_kpss, list_nms_scores

    def draw(self, srcimg, nms_bboxes, nms_kpss, nms_scores):
        for bbox, kps, score in zip(nms_bboxes, nms_kpss, nms_scores):
            # i = i[0]
            xmin, ymin, xamx, ymax = (
                int(bbox[0]),
                int(bbox[1]),
                int(bbox[2]),
                int(bbox[3]),
            )
            cv2.rectangle(srcimg, (xmin, ymin), (xamx, ymax), (0, 0, 255), thickness=2)
            for j in range(5):
                cv2.circle(srcimg, (int(kps[j, 0]), int(kps[j, 1])), 1, (0, 255, 0), thickness=-1)
            cv2.putText(
                srcimg,
                str(round(score, 3)),
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                thickness=1,
            )
        return srcimg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgpath", type=str, default="s_l.jpg", help="image path")
    parser.add_argument(
        "--onnxmodel",
        default="weights/scrfd_500m_kps.onnx",
        type=str,
        # choices=["weights/scrfd_500m_kps.onnx", "weights/scrfd_2.5g_kps.onnx", "weights/scrfd_10g_kps.onnx"],
        help="onnx model",
    )
    parser.add_argument("--confThreshold", default=0.5, type=float, help="class confidence")
    parser.add_argument("--nmsThreshold", default=0.5, type=float, help="nms iou thresh")

    parser.add_argument(
        "-v",
        "--verbose",
        dest="loglevel",
        help="set loglevel to INFO",
        action="store_const",
        default=logging.WARNING,
        const=logging.INFO,
    )
    parser.add_argument(
        "-vv",
        "--very-verbose",
        dest="loglevel",
        help="set loglevel to DEBUG",
        action="store_const",
        const=logging.DEBUG,
    )
    args = parser.parse_args()

    # initialize Log
    logging.basicConfig(
        format="[%(process)d,%(thread)x]%(asctime)s -%(levelname)s- %(name)s: %(message)s",
        level=args.loglevel,
    )

    mynet = SCRFD(args.onnxmodel, confThreshold=args.confThreshold, nmsThreshold=args.nmsThreshold)
    srcimg = cv2.imread(args.imgpath)
    list_nms_bboxes, list_nms_kpss, list_nms_scores = mynet.detect([srcimg])
    outimg = mynet.draw(srcimg, list_nms_bboxes[0], list_nms_kpss[0], list_nms_scores[0])

    winName = "Deep learning object detection in OpenCV"
    cv2.namedWindow(winName, 0)
    cv2.imshow(winName, outimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
