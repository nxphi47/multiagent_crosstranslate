import re
import argparse
import os
try:
    from langdetect import detect
    from polyglot.detect import Detector

    from polyglot.detect.base import logger as polyglot_logger
    polyglot_logger.setLevel("ERROR")
except ImportError as e:
    print('langdetect and other detector not installed, run the following:')
    print('bash install_mass_filter_noisy_prerequisite.sh')


"""
# This is copied from MASS paper github. https://github.com/microsoft/MASS/tree/master/MASS-unsupNMT
# TODO:
Insllation
* pyicu
apt-get update && apt-get install -y apt-transport-https && apt-get install  -y  libicu-dev
apt-get update && apt-get install -y apt-transport-https && apt-get install  -y  libicu-dev && ./setup.py install && pip install pyicu polyglot langdetect
pip install pyicu

*pycld2
git clone http://github.com/abosamoor/pycld2.git
cd pycld2
./setup.py install

*polyglot
pip install pyicu polyglot langdetect

* from here
git clone http://github.com/abosamoor/pycld2.git
cd pycld2
apt-get update && apt-get install -y apt-transport-https && apt-get install  -y  libicu-dev && ./setup.py install && pip install pyicu polyglot langdetect

# ------------------------------------------------------------------
"""

def get_parser():
    parser = argparse.ArgumentParser(description="Remove noisy data")

    parser.add_argument("--input", type=str,
                        help="The path of input file")
    parser.add_argument("--target", type=str, default=None,
                        help="The path of target file")

    parser.add_argument("--lang", type=str,
                        help="The language of input file")
    parser.add_argument("--lang_target", type=str, default=None,
                        help="The language of input file")

    parser.add_argument("--output", type=str, default=None,
                        help="The path of output file")
    parser.add_argument("--output_target", type=str, default=None,
                        help="The path of output file")

    parser.add_argument("--discard", type=str, default=None,
                        help="The path of discard file")
    parser.add_argument("--discard_target", type=str, default=None,
                        help="The path of discard file")



    return parser


def detect_exist_url(text):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    url1 = re.findall('http[s]?//(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    url2 = re.findall('http[s]? : / / (?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    url3 = re.findall('http[s]? / / (?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\), ]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls) > 0 or len(url1) > 0 or len(url2) > 0 or len(url3) > 0


def detect_lang(text, lang):
    try:
        for i, l in enumerate(Detector(text, quiet=True).languages):
            if l.code == lang and i == 0:
                return True
        if detect(text) == lang:
            return True
        return False
    except:
        return False


def run_single(args):
    if args.output is not None:
        f = open(args.output, 'w')
    if args.discard is not None:
        dis = open(args.discard, 'w')

    count = 0
    discard = 0
    allcount = 0
    with open(args.input, encoding='utf-8') as input_file:
        for line in input_file:
            allcount += 1
            line = line.strip()
            if detect_exist_url(line) is False:
                if detect_lang(line, args.lang) is True:
                    count += 1
                    if args.output is not None:
                        f.write(line + '\n')
                else:
                    discard += 1
                    if args.discard is not None:
                        dis.write(line + '\n')
            else:
                discard += 1
                if args.discard is not None:
                    dis.write(line + '\n')
                # print(line)
            if allcount % 100000 == 0:
                print("{} sentences processed, count: {}, discard: {}".format(allcount, count, discard))
    print(count, allcount)


def run_parallel(args):
    if args.output is not None:
        f = open(args.output, 'w')
    if args.discard is not None:
        dis = open(args.discard, 'w')
    if args.output_target is not None:
        f_target = open(args.output_target, 'w')
    if args.discard_target is not None:
        dis_target = open(args.discard_target, 'w')

    count = 0
    discard = 0
    allcount = 0
    with open(args.input, encoding='utf-8') as input_file:
        with open(args.target, encoding='utf-8') as target_file:
            for srcline, tgtline in zip(input_file, target_file):
                allcount += 1
                srcline = srcline.strip()
                tgtline = tgtline.strip()
                if not detect_exist_url(srcline) and not detect_exist_url(tgtline):
                    if detect_lang(srcline, args.lang) and detect_lang(tgtline, args.lang_target):
                        count += 1
                        if args.output is not None:
                            f.write(srcline + '\n')
                        if args.output_target is not None:
                            f_target.write(tgtline + '\n')
                    else:
                        discard += 1
                        if args.discard is not None:
                            dis.write(srcline + '\n')
                        if args.discard_target is not None:
                            dis_target.write(tgtline + '\n')
                else:
                    discard += 1
                    if args.discard is not None:
                        dis.write(srcline + '\n')
                    if args.discard_target is not None:
                        dis_target.write(tgtline + '\n')
                    # print(line)
                if allcount % 100000 == 0:
                    print("{} sentences processed, count: {}, discard: {}".format(allcount, count, discard))
    print(count, allcount)



def main():
    parser = get_parser()
    args = parser.parse_args()
    f = None
    dis = None
    f_target = None
    dis_target = None
    parallel = False
    if args.target is not None and os.path.exists(args.target):
        print('Require target....')
        assert args.lang_target is not None
        assert args.output_target is not None
        parallel = True

    print('Filtering data')
    print('Write to: {}'.format(args.output))
    print('Discard to: {}'.format(args.discard))
    if parallel:
        print('Write to target: {}'.format(args.output_target))
        print('Discard to target: {}'.format(args.discard_target))
        run_parallel(args)
    else:
        run_single(args)




if __name__ == "__main__":
    main()