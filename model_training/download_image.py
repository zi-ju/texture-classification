from better_bing_image_downloader import downloader

downloader("brick texture", limit=100, output_dir='dataset', adult_filter_off=True,
force_replace=False, timeout=60, filter="", verbose=True, badsites= [], name='Image')