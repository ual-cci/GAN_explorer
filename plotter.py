import cv2, os
import numpy as np
import reconnector

class Plotter(object):
    """
    Responsible for plotting experiments results
    """

    def __init__(self, renderer, getter):
        self.renderer = renderer
        self.getter = getter

        self.already_plotted = 0

        self.prepare_orders()

        self.font_multiplier = 1  # 1 with 1024
        #self.font_multiplier = 0.25  # with 256

    def prepare_orders(self):
        # Grid settings:
        # 5x10
        self.plot_row = list(np.arange(0, 2.0, 0.3)) + list(range(2,4+1))
        #self.target_tensors = ["16x16/Conv0_up/weight", "32x32/Conv0_up/weight", "64x64/Conv0_up/weight", "128x128/Conv0_up/weight", "256x256/Conv0_up/weight"]
        self.target_tensors = ["16x16/Conv0/weight", "32x32/Conv0/weight", "64x64/Conv0/weight", "128x128/Conv0/weight", "256x256/Conv0/weight"] # << Pre-trained PGGAN has these

        # 3x4
        self.plot_row = [0.0,0.5,1.0,4.0]
        # 3x6
        self.plot_row = [0.0,0.4,0.8,1.0,2.0,4.0]
        self.target_tensors = ["16x16/Conv0/weight", "64x64/Conv0/weight", "256x256/Conv0/weight"]

        ## args.model_path = 'models/karras2018iclr-celebahq-1024x1024.pkl'
        ##self.target_tensors = ["16x16/Conv0/weight", "64x64/Conv0/weight", "256x256/Conv0/weight"] # << Pre-trained PGGAN has these
        ##self.plot_row = [0.0,0.4,0.8,1.0,2.0,4.0] # << Pre-trained PGGAN has these

        ## args.model_path = 'models/karras2018iclr-lsun-car-256x256.pkl'
        self.target_tensors = ["16x16/Conv0/weight", "64x64/Conv0/weight", "128x128/Conv0/weight"] # << Pre-trained PGGAN with 256x256 resolution
        self.plot_row = [0.0,0.4,0.8,1.0,2.0]



    def prepare_with_set_tensors(self):
        # these are prepared once to allow for smooth anim!
        self.tensor2fixed_order = {}
        net = self.getter.serverside_handler._Gs # < ProgressiveGAN_Handler._Gs
        for tensor_name in self.target_tensors:
            res = reconnector.dgb_get_res(net, tensor_name)
            print("tensor_name 2 res", tensor_name, res)

            go_up_to_muliples = int(self.plot_row[-1])
            # from 0 ... res
            randomized_list = np.arange(res)
            np.random.shuffle(randomized_list)
            #randomized_list = np.random.choice(list(range(res)), res, replace=False)
            OVERALL_ORDER = randomized_list
            for i in range(go_up_to_muliples):
                randomized_list = np.arange(res)
                np.random.shuffle(randomized_list)
                #randomized_list = np.random.choice(list(range(res)), res, replace=False)
                OVERALL_ORDER = np.append(OVERALL_ORDER, randomized_list)

            self.tensor2fixed_order[ tensor_name ] = OVERALL_ORDER

        print("debug self.plot_row", self.plot_row)
        print("debug self.tensor2fixed_order.keys", self.tensor2fixed_order.keys())
        for tensor_name in self.target_tensors:
            print("debug self.tensor2fixed_order[",tensor_name,"]=> len, min, max : ", len(self.tensor2fixed_order[tensor_name]), min(self.tensor2fixed_order[tensor_name]), max(self.tensor2fixed_order[tensor_name]))

    def plot(self, current_point, counter_override = -1):
        print("plotter called!")
        # 1 plot grid
        #"""
        if counter_override != -1:
            self.already_plotted = counter_override

        self.all_rows(current_point)
        #self.one_row_effectStrength_of_reconnector(current_point)
        #"""

        # 2 plot animation
        """
        self.animate_effect(current_point, self.target_tensors[0])
        """

    # Plotting grids:

    def all_rows(self,current_point):

        rows = []

        for target in self.target_tensors:
            name = str(self.already_plotted).zfill(3)+"_plot_"+target.split("/")[0]+".png"
            row = self.one_row_effectStrength_of_reconnector(current_point, target, all_name=name, plot_row=self.plot_row)
            rows.append(row)


        full_image = self.concatenate_images_v(rows)
        folder = "renders/PLOTS_FULL/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        cv2.imwrite(folder+str(self.already_plotted).zfill(3)+"__FULL_IMAGE"+".png", full_image)

        self.already_plotted += 1


    def one_row_effectStrength_of_reconnector(self, current_point, tensor_name = "16x16/Conv0_up/weight", all_name = "all.png",
                                              save_individuals=False, save_rows=False, plot_row = [0,0.5,1.0,2.0]):
        print("Plotting row for",tensor_name)
        # init
        net = self.getter.serverside_handler._Gs # < ProgressiveGAN_Handler._Gs

        # prepare the ordering
        res = reconnector.dgb_get_res(net, tensor_name)
        print("got res value as", res)

        OVERALL_ORDER = self.tensor2fixed_order[ tensor_name ] # these were prepared once to allow for smooth anim!

        print("OVERALL_ORDER", len(OVERALL_ORDER))
        print("debug OVERALL_ORDER", len(OVERALL_ORDER), min(OVERALL_ORDER), max(OVERALL_ORDER))
        #print("debug OVERALL_ORDER", OVERALL_ORDER)

        print("plot_row", len(plot_row), " : ", plot_row)

        images = []
        names = []

        for value_to_select in plot_row:
            # 1.0 == res
            end_val = int(value_to_select * res)
            FIXED_ORDER = OVERALL_ORDER[:end_val]

            edited_net = reconnector.reconnect_DIRECT_ORDER(net, FIXED_ORDER, tensor_name)

            # generate image!
            latents = np.asarray([current_point])
            image = self.getter.latent_to_image_localServerSwitch(latents)

            # Add text?
            include_texts = True
            if include_texts:
                font = cv2.FONT_HERSHEY_SIMPLEX
                bottomRightCornerOfText = (int((1024 - 100 - 25) * self.font_multiplier), int((1024 - 25) * self.font_multiplier)) # usually 16x16 to 256x256
                topLeftCornerOfText = (int(25 * self.font_multiplier), int((75+5) * self.font_multiplier)) # usually 0.0 to 4.0
                #bottomLeftCornerOfText = (10, 1000)
                fontScale = 2 * self.font_multiplier
                fontColor = (255, 255, 255)
                lineThickness = max(int(3 * self.font_multiplier),1)

                cv2.putText(image, str(value_to_select),
                            bottomRightCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            lineThickness)

                if value_to_select == 0.0:
                    n = tensor_name.split("/")[0]
                    cv2.putText(image, n,
                                topLeftCornerOfText,
                                font,
                                fontScale,
                                fontColor,
                                lineThickness)

            images.append(image)
            names.append("_effect-"+str(value_to_select))

            # Restart the network afterwards!
            net = reconnector.restore_net(edited_net)

        """
        if save_individuals:
            # Save images as a row:
            print("created", len(images), "images!")
            saved_already = 0
            for i, IMG in enumerate(images):
                message = "Saved file"
                folder = "DBG/"
                if not os.path.exists(folder):
                    os.mkdir(folder)
                filename = folder+"saved_" + str(saved_already).zfill(4) + names[i] + ".png"
                saved_already += 1
                print("Saving in good quality as ", filename)
                cv2.imwrite(filename, IMG)
            print("SAVED ALL")
        """

        row = self.concatenate_images_h(images)

        if save_rows:
            folder = "renders/PLOTS/"
            if not os.path.exists(folder):
                os.mkdir(folder)
            cv2.imwrite(folder+all_name, row)

        return row

    # Plotting animations:
    def animate_effect(self, current_point, tensor_name = "16x16/Conv0_up/weight"):
        print("Plotting animation for",tensor_name)
        # init
        net = self.getter.serverside_handler._Gs # < ProgressiveGAN_Handler._Gs

        # prepare the ordering
        res = reconnector.dgb_get_res(net, tensor_name)
        print("got res value as", res)

        OVERALL_ORDER = self.tensor2fixed_order[ tensor_name ] # these were prepared once to allow for smooth anim!

        print("OVERALL_ORDER", len(OVERALL_ORDER))
        print("debug OVERALL_ORDER", len(OVERALL_ORDER), min(OVERALL_ORDER), max(OVERALL_ORDER))
        #print("debug OVERALL_ORDER", OVERALL_ORDER)

        plot_row = list(np.arange(0.0, 2.0, 0.01)) # + list(np.arange(2.0, 0.0, 0.1))
        print("plot_row", len(plot_row), " : ", plot_row)

        images = []
        names = []

        """
        for value_to_select in plot_row:
            # 1.0 == res
            end_val = int(value_to_select * res)
            FIXED_ORDER = OVERALL_ORDER[:end_val]
        """
        folder = "renders/ANIMATION/"
        if not os.path.exists(folder):
            os.mkdir(folder)
        saved_already = 0

        # real smooth turning on!
        for end_val in range(0, 1024, 2): #range(len(OVERALL_ORDER))
            # 1.0 == res
            #end_val = int(value_to_select * res)
            FIXED_ORDER = OVERALL_ORDER[:end_val]
            print("--selected 0 to ",end_val, "getting in total", len(FIXED_ORDER), "of numbers from which we will create pairs")

            edited_net = reconnector.reconnect_DIRECT_ORDER(net, FIXED_ORDER, tensor_name)

            # generate image!
            latents = np.asarray([current_point])
            image = self.getter.latent_to_image_localServerSwitch(latents)

            #images.append(image)
            name = "_effect-"+str(end_val)

            filename = folder + "saved_" + str(saved_already).zfill(4) + name + ".png"
            saved_already += 1
            print("Saving in good quality as ", filename)
            cv2.imwrite(filename, image)

            # Restart the network afterwards!
            net = reconnector.restore_net(edited_net)

        print("SAVED ALL")

    # Helper functions

    def concatenate_images_h(self, images):
        i = np.asarray(images)
        print("images.shape", i.shape)
        hstack = np.hstack(i)
        print("hstack.shape", hstack.shape)
        return hstack

    def concatenate_images_v(self, images):
        i = np.asarray(images)
        print("images.shape", i.shape)
        vstack = np.vstack(i)
        print("vstack.shape", vstack.shape)
        return vstack
