from jinja2 import Environment, FileSystemLoader, select_autoescape
import pathlib
from .compiler_context import CompilerContext

def render_launcher(comp_ctx: CompilerContext):
    this_file_dir = pathlib.Path(__file__).parent
    env = Environment(
        loader=FileSystemLoader(this_file_dir),
        autoescape=select_autoescape(),
    )
    template = env.get_template("launcher.cpp.jinja")
    return template.render(comp_ctx=comp_ctx, mod_name = comp_ctx.name)
