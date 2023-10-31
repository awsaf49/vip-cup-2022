import os
import tensorflow as tf
from tensorflow import keras


def reload_model_weights(model, pretrained_dict, sub_release, pretrained="imagenet", mismatch_class=None, request_resolution=-1, method="nearest"):
    if pretrained is None:
        return
    if isinstance(pretrained, str) and pretrained.endswith(".h5"):
        print(">>>> Load pretrained from:", pretrained)
        # model.load_weights(pretrained, by_name=True, skip_mismatch=True)
        load_weights_with_mismatch(model, pretrained, mismatch_class, request_resolution, method)
        return pretrained

    file_hash = pretrained_dict.get(model.name, {}).get(pretrained, None)
    if file_hash is None:
        print(">>>> No pretrained available, model will be randomly initialized")
        return None

    if isinstance(file_hash, dict):
        # file_hash is a dict like {224: "aa", 384: "bb", 480: "cc"}
        if request_resolution == -1:
            input_height = model.input_shape[1]
            if input_height is None:  # input_shape is (None, None, 3)
                request_resolution = max(file_hash.keys())
            else:
                request_resolution = min(file_hash.keys(), key=lambda ii: abs(ii - input_height))
        pretrained = "{}_".format(request_resolution) + pretrained
        file_hash = file_hash[request_resolution]
        # print(f"{request_resolution = }, {pretrained = }, {file_hash = }")
    elif request_resolution == -1:
        request_resolution = 224  # Default is 224

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/{}/{}_{}.h5"
    url = pre_url.format(sub_release, model.name, pretrained)
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models", file_hash=file_hash)
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return None
    else:
        print(">>>> Load pretrained from:", pretrained_model)
        # model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
        load_weights_with_mismatch(model, pretrained_model, mismatch_class, request_resolution, method)
        return pretrained_model


def load_weights_with_mismatch(model, weight_file, mismatch_class=None, request_resolution=-1, method="nearest"):
    model.load_weights(weight_file, by_name=True, skip_mismatch=True)
    input_height, input_width = model.input_shape[1], model.input_shape[2]
    # if mismatch_class is not None:
    if mismatch_class is not None and (request_resolution != input_height or request_resolution != input_width):
        try:
            import h5py

            print(">>>> Reload mismatched weights: {} -> {}".format(request_resolution, (input_height, input_width)))
            ff = h5py.File(weight_file, mode="r")
            weights = ff["model_weights"]
            for ii in model.layers:
                if isinstance(ii, mismatch_class) and ii.name in weights:
                    print(">>>> Reload layer:", ii.name)
                    ss = weights[ii.name]
                    # ss = {ww.decode().split("/")[-1] : tf.convert_to_tensor(ss[ww]) for ww in ss.attrs['weight_names']}
                    ss = {ww.decode("utf8") if hasattr(ww, "decode") else ww: tf.convert_to_tensor(ss[ww]) for ww in ss.attrs["weight_names"]}
                    ss = {kk.split("/")[-1]: vv for kk, vv in ss.items()}
                    model.get_layer(ii.name).load_resized_pos_emb(ss, method=method)
            ff.close()

        # print(">>>> Reload mismatched PositionalEmbedding weights: {} -> {}".format(request_resolution, input_shape[0]))
        # bb = keras.models.load_model(pretrained_model)
        # for ii in model.layers:
        #     if isinstance(ii, mismatch_class):
        #         print(">>>> Reload layer:", ii.name)
        #         model.get_layer(ii.name).load_resized_pos_emb(bb.get_layer(ii.name))
        except:
            print("[Error] something went wrong in load_weights_with_mismatch")
            pass


def state_dict_stack_by_layer(state_dict, skip_weights=["num_batches_tracked"], unstack_weights=[]):
    stacked_state_dict = {}
    for kk, vv in state_dict.items():
        split_kk = kk.split(".")
        vv = vv.numpy() if hasattr(vv, "numpy") else vv
        if split_kk[-1] in skip_weights:
            continue

        if split_kk[-1] in unstack_weights:
            stacked_state_dict[kk] = [vv]
        else:
            # split_kk[-1] in ["weight", "bias", "running_mean", "running_var", "gain"]
            layer_name = ".".join(split_kk[:-1])
            stacked_state_dict.setdefault(layer_name, []).append(vv)
    return stacked_state_dict


def match_layer_names_with_torch(target_names, tail_align_dict={}, full_name_align_dict={}, tail_split_position=2):
    layer_names_matched_torch = [""] * len(target_names)
    raw_id_dict = {ii: id for id, ii in enumerate(target_names)}

    # is_tail_align_dict_split_by_stack = len(tail_align_dict) > 0 and isinstance(list(tail_align_dict.values())[0], dict)
    for id, ii in enumerate(target_names):
        name_split = ii.split("_")
        stack_name = name_split[0]
        head_name = "_".join(name_split[:tail_split_position])
        tail_name = "_".join(name_split[tail_split_position:])
        # cur_tail_align_dict = tail_align_dict[stack_name] if is_tail_align_dict_split_by_stack else tail_align_dict
        cur_tail_align_dict = tail_align_dict.get(stack_name, tail_align_dict)
        # print("id = {}, ii = {}, stack_name = {}, tail_name = {}".format(id, ii, stack_name, tail_name))
        if ii in full_name_align_dict:
            align = full_name_align_dict[ii]
            if isinstance(align, str):
                align = raw_id_dict[align]
            else:
                align = id + align if align < 0 else align
            layer_names_matched_torch.insert(align, ii)
            layer_names_matched_torch.pop(-1)
        elif tail_name in cur_tail_align_dict:
            align = cur_tail_align_dict[tail_name]
            if isinstance(align, str):
                align = raw_id_dict[head_name + "_" + align]
            else:
                align = id + align if align < 0 else align
            layer_names_matched_torch.insert(align, ii)
            layer_names_matched_torch.pop(-1)
        else:
            layer_names_matched_torch[id] = ii
    return layer_names_matched_torch


def align_layer_names_multi_stage(target_names, tail_align_dict={}, full_name_align_dict={}, tail_split_position=2, specific_match_func=None):
    if isinstance(tail_split_position, int):
        tail_align_dict, full_name_align_dict, tail_split_position = [tail_align_dict], [full_name_align_dict], [tail_split_position]
    full_name_align_dict = full_name_align_dict if isinstance(full_name_align_dict, (list, tuple)) else [full_name_align_dict]
    # for ii, jj, kk in zip(tail_align_dict, full_name_align_dict, tail_split_position):
    for idx in range(max(len(tail_split_position), len(full_name_align_dict))):
        cur_tail = tail_align_dict[idx] if idx < len(tail_align_dict) else {}
        cur_full = full_name_align_dict[idx] if idx < len(full_name_align_dict) else {}
        cur_split = tail_split_position[idx] if idx < len(tail_split_position) else tail_split_position[-1]
        print(">>>> tail_align_dict:", cur_tail)
        print(">>>> full_name_align_dict:", cur_full)
        print(">>>> tail_split_position:", cur_split)
        target_names = match_layer_names_with_torch(target_names, cur_tail, cur_full, cur_split)

    if specific_match_func is not None:
        target_names = specific_match_func(target_names)
    return target_names


def keras_reload_stacked_state_dict(model, stacked_state_dict, layer_names_matched_torch, additional_transfer={}, save_name=None):
    import numpy as np

    for kk, tf_layer_name in zip(stacked_state_dict.keys(), layer_names_matched_torch):
        print("  torch layer name: {}, tf layer name: {}".format(kk, tf_layer_name))
        tf_layer = model.get_layer(tf_layer_name)
        tf_weights = tf_layer.get_weights()
        torch_weight = stacked_state_dict[kk]
        print("    Before: [{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        if isinstance(tf_layer, keras.layers.DepthwiseConv2D):  # DepthwiseConv2D is instance of Conv2D, put it first
            torch_weight[0] = np.transpose(torch_weight[0], (2, 3, 0, 1))
        elif isinstance(tf_layer, keras.layers.Conv2D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 3, 1, 0))
            if len(torch_weight) > 2:  # gain
                torch_weight[2] = np.squeeze(torch_weight[2])
        elif isinstance(tf_layer, keras.layers.PReLU):
            torch_weight[0] = np.expand_dims(np.expand_dims(torch_weight[0], 0), 0)
        elif isinstance(tf_layer, keras.layers.Conv1D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 1, 0))
        elif isinstance(tf_layer, keras.layers.Dense):
            # [Note] if it's fc layer after flatten, weights need to reshape according to NCHW --> NHWC
            torch_weight[0] = torch_weight[0].T

        for add_layer, add_transfer in additional_transfer.items():
            if isinstance(add_layer, str):
                if tf_layer.name == add_layer:
                    torch_weight = add_transfer(torch_weight)
            elif isinstance(tf_layer, add_layer):
                torch_weight = add_transfer(torch_weight)
        print("    After: [{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        tf_layer.set_weights(torch_weight)

    if save_name is None:
        save_name = model.name + ".h5"
    if len(save_name) != 0:
        print()
        print(">>>> Save model to:", save_name)
        model.save(save_name)


def try_save_pth_and_onnx(torch_model, save_pth=True, save_onnx=True, input_shape=(224, 224)):
    import torch

    input_shape = input_shape[:2]
    if save_pth:
        try:
            output_name = torch_model.__class__.__name__ + ".pth"
            traced_cell = torch.jit.trace(torch_model, (torch.randn(10, 3, *input_shape)))
            torch.jit.save(traced_cell, output_name)
            print(">>>> Saved to:", output_name)
        except:
            print(">>>> Saved pth failed")
            pass

    if save_onnx:
        try:
            output_name = torch_model.__class__.__name__ + ".onnx"
            torch.onnx.export(
                model=torch_model,
                args=torch.randn(10, 3, *input_shape),
                f=output_name,
                verbose=False,
                keep_initializers_as_inputs=True,
                training=torch.onnx.TrainingMode.PRESERVE,
                do_constant_folding=False,
                opset_version=13,
            )
            print(">>>> Saved to:", output_name)
        except:
            print(">>>> Saved onnx failed")
            pass


def keras_reload_from_torch_model(
    torch_model,  # Torch model, Torch state_dict wights or Torch model weights file
    keras_model=None,
    input_shape=(224, 224),
    skip_weights=["num_batches_tracked"],
    unstack_weights=[],
    tail_align_dict={},
    full_name_align_dict={},
    tail_split_position=2,
    additional_transfer={},
    specific_match_func=None,
    save_name=None,
    do_convert=True,
):
    import torch
    import numpy as np
    import tensorflow as tf
    from keras_cv_attention_models import test_images

    input_shape = input_shape[:2]
    if isinstance(torch_model, str):
        print(">>>> Reload Torch weight file:", torch_model)
        torch_model = torch.load(torch_model, map_location=torch.device("cpu"))
        torch_model = torch_model.get("model", torch_model.get("state_dict", torch_model))
    is_state_dict = isinstance(torch_model, dict)

    """ Chelsea the cat  """
    img = test_images.cat()
    img = keras.applications.imagenet_utils.preprocess_input(tf.image.resize(img, input_shape), mode="torch").numpy()

    if not is_state_dict:
        _ = torch_model.eval()
        state_dict = torch_model.state_dict()

        try:
            # from torchsummary import summary
            # summary(torch_model, (3, *input_shape), device="cpu")
            """Torch Run predict"""
            out = torch_model(torch.from_numpy(np.expand_dims(img.transpose(2, 0, 1), 0).astype("float32")))
            out = out.detach().cpu().numpy()
            # out = tf.nn.softmax(out).numpy()  # If classifier activation is not softmax
            torch_out = keras.applications.imagenet_utils.decode_predictions(out)
        except:
            pass
    else:
        state_dict = torch_model

    """ Convert torch weights """
    # torch_params = {kk: (np.cumproduct(vv.shape)[-1] if len(vv.shape) != 0 else 1) for kk, vv in state_dict.items() if ".num_batches_tracked" not in kk}
    stacked_state_dict = state_dict_stack_by_layer(state_dict, skip_weights=skip_weights, unstack_weights=unstack_weights)
    print(">>>> torch_model total_parameters :", np.sum([np.sum([np.prod(jj.shape) for jj in ii]) for ii in stacked_state_dict.values()]))
    aa = {kk: [1 if isinstance(jj, float) else jj.shape for jj in vv] for kk, vv in stacked_state_dict.items()}
    print(">>>> Torch weights:")
    _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
    print()

    if keras_model is None:
        return

    """ Keras model weights """
    target_names = [ii.name for ii in keras_model.layers if len(ii.weights) != 0]
    aa = {keras_model.get_layer(ii).name: [jj.shape.as_list() for jj in keras_model.get_layer(ii).weights] for ii in target_names}
    print(">>>> keras_model total_parameters :", np.sum([np.sum([np.prod(jj) for jj in ii]) for ii in aa.values()]))
    print(">>>> Keras weights:")
    _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
    print()

    """ Load torch weights and save h5 """
    if len(tail_align_dict) != 0 or len(full_name_align_dict) != 0 or specific_match_func is not None:
        aligned_names = align_layer_names_multi_stage(target_names, tail_align_dict, full_name_align_dict, tail_split_position, specific_match_func)
        aa = {keras_model.get_layer(ii).name: [jj.shape.as_list() for jj in keras_model.get_layer(ii).weights] for ii in aligned_names}
        print(">>>> Keras weights matched torch:")
        _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
        print()
    else:
        aligned_names = target_names

    if not do_convert:
        return

    save_name = save_name if save_name is not None else keras_model.name + "_imagenet.h5"
    print(">>>> Keras reload torch weights:")
    keras_reload_stacked_state_dict(keras_model, stacked_state_dict, aligned_names, additional_transfer, save_name=save_name)
    print()

    """ Keras run predict """
    try:
        pred = keras_model(tf.expand_dims(img, 0)).numpy()
        # pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
        print(">>>> Keras model prediction:", keras.applications.imagenet_utils.decode_predictions(pred)[0])
        print()
        if not is_state_dict:
            print(">>>> Torch model prediction:", torch_out)
    except:
        pass
