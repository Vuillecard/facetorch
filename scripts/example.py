import hydra
from facetorch import FaceAnalyzer
from omegaconf import DictConfig
import time 
import os
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    analyzer = FaceAnalyzer(cfg.analyzer)
    
    # check if it is a directory
    img_count = 0
    start = time.time()
    
    if os.path.isdir(cfg.path_image):
        print('is a directory')
        os.makedirs(cfg.path_output, exist_ok=True)
        for root, dirs, files in os.walk(cfg.path_image):
            for file in files:
                response = analyzer.run(
                    image_source=os.path.join(root, file),
                    batch_size=cfg.batch_size,
                    fix_img_size=cfg.fix_img_size,
                    return_img_data=cfg.return_img_data,
                    include_tensors=cfg.include_tensors,
                    path_output=os.path.join(cfg.path_output, file)
                )
                img_count += 1
                
    else:
        response = analyzer.run(
            image_source=cfg.path_image,
            batch_size=cfg.batch_size,
            fix_img_size=cfg.fix_img_size,
            return_img_data=cfg.return_img_data,
            include_tensors=cfg.include_tensors,
            path_output=cfg.path_output,
        )
        img_count += 1
        print(response)
    # if cfg.path_image
    # response = analyzer.run(
    #     image_source=cfg.path_image,
    #     batch_size=cfg.batch_size,
    #     fix_img_size=cfg.fix_img_size,
    #     return_img_data=cfg.return_img_data,
    #     include_tensors=cfg.include_tensors,
    #     path_output=cfg.path_output,
    # )
    # print(response)

    # start = time.time()
    # for i in range(5):
    #     response = analyzer.run(
    #     image_source=cfg.path_image,
    #     batch_size=cfg.batch_size,
    #     fix_img_size=cfg.fix_img_size,
    #     return_img_data=cfg.return_img_data,
    #     include_tensors=cfg.include_tensors,
    #     path_output=cfg.path_output,
    # )
    end = time.time()
    print(f'Average time per img: {(end-start)/img_count} seconds')

if __name__ == "__main__":
    main()
