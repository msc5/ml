from rich.console import Console
from rich.theme import Theme

THEME = Theme({
    'dot.title': 'cyan',
    'dot.border': 'black',
    'dot.key': 'yellow',
    'dot.type': 'blue',
    'dot.complete': 'green',
    'dot.missing': 'red underline',
    'dot.important': 'magenta',

    'bar.complete': 'green',
    'bar.finished': 'green',
    'progress.percentage': 'cyan',
})


console = Console(theme=THEME)
