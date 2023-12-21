from rich.progress import (Progress, BarColumn, TextColumn, TimeElapsedColumn, 
                           TimeRemainingColumn, MofNCompleteColumn, SpinnerColumn, 
                           TransferSpeedColumn) 


def get_progress():
    progress = Progress(
                SpinnerColumn(),
                MofNCompleteColumn(),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                TransferSpeedColumn(),
                TextColumn("[progress.description]{task.description}"),
                )
    return progress
