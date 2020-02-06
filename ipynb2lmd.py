# This scripts is created by Luigi Ballabio: https://github.com/lballabio/ipynb2lmd

import io
import json
import os
import re
import sys


def join(lines):
    return "".join(lines)


def in_prompt(prompt_number):
    return "In [%d]: " % prompt_number


def out_prompt(prompt_number):
    return "Out[%d]: " % prompt_number


def add_prompt(lines, prompt):
    "add the prompt on the first line, indent all other lines accordingly"
    indentation = " " * len(prompt)
    return [prompt + lines[0]] + [indentation + l for l in lines[1:]]


def indent(lines):
    "add indentation required for code samples in Markdown"
    return ["    " + l for l in lines]


def code(lines):
    return join(indent(lines))


formulas = re.compile(r"(\$\$?)([^\$]+)(\$\$?)")


def replace_formulas(text):
    "In Leanpub Markdown, formulas are delimited by {$$}...{/$$}"
    return formulas.sub(r"{$$}\2{/$$}", text)


def text(lines):
    return replace_formulas(join(lines))


def convert_markdown(cell, out):
    content = text(cell["source"])
    if content.startswith("#"):
        # a heading
        out.write(u"\n")
    out.write(content)
    out.write(u"\n\n")


def convert_raw(cell, out):
    out.write(join(cell["source"]))
    out.write(u"\n\n")


def convert_code(cell, out, base_name, output_dir):
    prompt_number = cell["execution_count"]
    if cell["source"]:
        out.write(code(add_prompt(cell["source"], in_prompt(prompt_number))))
    out.write(u"\n")
    last_output_type = None
    for output in cell["outputs"]:
        output_type = output["output_type"]
        if output_type == "execute_result":
            convert_result(
                output, out, prompt_number, continued=(output_type == last_output_type)
            )
        elif output_type == "stream":
            convert_stream(
                output, out, prompt_number, continued=(output_type == last_output_type)
            )
        elif output_type == "error":
            convert_error(output, out, prompt_number)
        elif output_type == "display_data":
            if last_output_type in ["execute_result", "stream"]:
                out.write(u"\n\n")
            convert_image(output, out, base_name, output_dir, prompt_number)
        else:
            raise Exception("unknown output type: %s" % output_type)
        last_output_type = output_type
    if last_output_type in ["execute_result", "stream"] and not (
        "data" in output and "text/html" in output["data"]
    ):
        out.write(u"\n\n")
    out.write(u"\n")


def convert_result(output, out, prompt_number, continued=False):
    out.write(u"    \n")
    if "data" in output and "text/html" in output["data"]:
        if not continued:
            out.write(code(add_prompt([u""], out_prompt(prompt_number))))
            out.write(u"\n")
        convert_html(join(output["data"]["text/html"]), out)
    else:
        prompt = out_prompt(prompt_number)
        if continued:
            # we don't want the prompt, but we need to indent as if it
            # was there.
            prompt = " " * len(prompt)
        out.write(code(add_prompt(output["data"]["text/plain"], prompt)))


def convert_stream(output, out, prompt_number, continued=False):
    out.write(u"    \n")
    prompt = out_prompt(prompt_number)
    if continued:
        # we don't want the prompt, but we need to indent as if it
        # was there.
        prompt = " " * len(prompt)
    out.write(code(add_prompt(output["text"], prompt)))


table_html = re.compile(r"<table.*?>(.*)</table>", re.DOTALL)


def convert_html(html, out):
    match = table_html.search(html)
    if match:
        convert_table(match.group(1), out)
    else:
        raise Exception("Unknown html: %s" % html)


row_html = re.compile(r"<tr.*?>(.*?)</tr>", re.DOTALL)
cell_html = re.compile(r"<t[dh].*?>(.*?)</t[dh]>", re.DOTALL)


def convert_table(table, out):
    data = []
    rows = row_html.findall(table)
    for r in rows:
        data.append([x.strip() for x in cell_html.findall(r)])

    widths = [max(len(d[i]) for d in data) for i in range(len(data[0]))]
    format = "|" + "|".join([" %%%ds " % w for w in widths]) + "|\n"

    total_width = len(format % tuple("" for e in widths))
    if total_width <= 60:
        width = "narrow"
    elif total_width >= 80:
        width = "wide"
    else:
        width = "default"
    out.write(u'\n{width="%s"}\n' % width)

    out.write(format % tuple(data[0]))
    out.write("|" + "|".join([u"-" * (w + 2) for w in widths]) + "|\n")
    for d in data[1:]:
        out.write(format % tuple(d))
    out.write(u"\n\n")


terminal_codes = re.compile(r".\[[01](;\d\d)?m")


def convert_error(output, out, prompt_number):
    def unescape_terminal_codes(line):
        return terminal_codes.sub("", line)

    out.write(u"    \n")
    # There are embedded \n in the lines...
    lines = [l + "\n" for line in output["traceback"] for l in line.split("\n")]
    # ...and control codes for the terminal
    out.write(
        code(
            add_prompt(
                [unescape_terminal_codes(l) for l in lines], out_prompt(prompt_number)
            )
        )
    )
    out.write(u"\n\n")


def convert_image(output, out, base_name, output_dir, prompt_number):
    ext = extension(output)
    images_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    image_name = "%s-%d.%s" % (base_name.replace(" ", "_"), prompt_number, ext)
    image_path = os.path.join(images_dir, image_name)
    with open(image_path, "w") as image:
        image.write(output["data"]["image/%s" % ext].decode("base64"))
    out.write(u"\n")
    out.write(u"![](images/%s)" % image_name)
    out.write(u"\n\n")


def extension(output):
    candidates = set(output["data"].keys()) - {"text/plain"}
    # whatever key remains should be the extension
    if len(candidates) > 1:
        raise Exception("multiple extensions found: %s" % candidates)
    candidate = str(candidates.pop())
    if not candidate.startswith("image/"):
        raise Exception("not an image type: %s" % candidate)
    return candidate[6:]


def convert(path, output_dir):
    _, filename = os.path.split(path)
    base_name, _ = os.path.splitext(filename)
    base_name = base_name.lower()
    md_name = base_name + ".md"

    with open(path) as f:
        data = json.load(f)
    cells = data["cells"]

    with io.open(os.path.join(output_dir, md_name), "w") as out:
        for cell in cells:
            cell_type = cell["cell_type"]
            if cell_type == "markdown":
                convert_markdown(cell, out)
            elif cell_type == "code":
                convert_code(cell, out, base_name, output_dir)
            elif cell_type == "raw":
                convert_raw(cell, out)
            else:
                raise Exception("unknown cell type: %s" % cell_type)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            """
    Usage: %s notebook.ipynb output_dir

    The file notebook.md will be created in output_dir; if the
    notebook contains images, they will be extracted and stored
    in the output_dir/images folder.
        """
            % sys.argv[0]
        )
        sys.exit(1)

    convert(os.path.abspath(sys.argv[1]), sys.argv[2])
