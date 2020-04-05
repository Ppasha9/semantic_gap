import os
import cv2

import matplotlib.pyplot as plt


class ChairWidthCalculator(object):
    @staticmethod
    def find_chair_width(data_img, templates_imgs, hyperparameters, logger=None, img_name=None, output_dir=None, verbose=True):
        orb_features_num = hyperparameters["num_orbs_features"]
        orb_match_dist = hyperparameters["orb_match_dist"]
        chair_height_coeff = hyperparameters["chair_height_coeff"]

        max_num_of_good_matches = 0
        kp_src_best = None
        kp_dst_best = None
        best_matches = None
        index_of_best_template = -1

        # ==== FIND BEST MATCHES ====

        dst_img = cv2.cvtColor(data_img, cv2.COLOR_BGR2GRAY)

        for template_ind, cur_template_img in enumerate(templates_imgs):
            if verbose:
                logger.info("============Processing template: %d==============" % (template_ind + 1))

            src_img = cv2.cvtColor(cur_template_img, cv2.COLOR_BGR2GRAY)

            orb_detector = cv2.ORB_create(nfeatures=orb_features_num)
            kp_src, des_src = orb_detector.detectAndCompute(src_img, None)
            kp_dst, des_dst = orb_detector.detectAndCompute(dst_img, None)

            matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = matcher.match(des_src, des_dst)

            good_matches = list(filter(lambda match: match.distance <= orb_match_dist, matches))
            cur_matches_num = len(good_matches)

            if verbose:
                logger.info("Current num of good matches: %d" % cur_matches_num)

            if max_num_of_good_matches < cur_matches_num:
                max_num_of_good_matches = cur_matches_num
                kp_src_best = kp_src
                kp_dst_best = kp_dst
                best_matches = good_matches
                index_of_best_template = template_ind

            if verbose:
                logger.info("============           Done.           ==============")

        if verbose:
            logger.info("Best matches for image %s were found by template %d. Num of matches: %d"
                        % (img_name, index_of_best_template + 1, len(best_matches)))

        best_src = cv2.cvtColor(templates_imgs[index_of_best_template], cv2.COLOR_BGR2GRAY)
        img_matches = cv2.drawMatches(best_src, kp_src_best,
                                      dst_img, kp_dst_best,
                                      best_matches,
                                      None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        if verbose:
            path = os.path.normpath(os.path.join(output_dir, "matches_%s.png" % img_name))
            plt.figure(figsize=(30, 30))
            plt.imshow(img_matches, interpolation="nearest")
            plt.savefig(path)
            plt.close()
            logger.info("Best matches plot saved by path: %s" % path)

        # ==== FIND CHAIR WIDTH ====

        top_edge_of_chair_y = (1 - chair_height_coeff) * data_img.shape[0]
        x_coords = []
        for match in best_matches:
            x, y = kp_dst_best[match.trainIdx].pt

            if y > top_edge_of_chair_y:
                x_coords.append(x)
        x_coords.sort()
        width = (x_coords[-1] - x_coords[0]) / data_img.shape[1]

        if verbose:
            logger.info("Found width of chair: %.4f" % width)

        return width
