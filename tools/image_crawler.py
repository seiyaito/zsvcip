import argparse
import os
import logging

from icrawler.builtin import BingImageCrawler

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_dir", type=str, default="images")
    parser.add_argument("-j", "--num_threads", type=int, default=4)
    parser.add_argument("-k", "--keyword", type=str, default="cat")
    parser.add_argument("-n", "--max_num", type=int, default=10)
    parser.add_argument("--license", type=str, default="creativecommons")

    args = parser.parse_args()

    os.makedirs(args.root_dir, exist_ok=True)

    filters = {
        "size": "large",
        "license": args.license,
    }

    bing_crawler = BingImageCrawler(
        downloader_threads=args.num_threads,
        storage={"root_dir": args.root_dir}, 
    )

    fh = logging.FileHandler(os.path.join(args.root_dir, "log.txt"))
    fh.setLevel(logging.DEBUG)
    fh_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s - %(name)s - %(funcName)s - %(message)s')
    fh.setFormatter(fh_formatter)
    
    bing_crawler.logger.addHandler(fh)
    bing_crawler.feeder.logger.addHandler(fh)
    bing_crawler.parser.logger.addHandler(fh)
    bing_crawler.downloader.logger.addHandler(fh)

    bing_crawler.crawl(keyword=args.keyword, filters=filters, max_num=args.max_num) 
