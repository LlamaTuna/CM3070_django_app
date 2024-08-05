import cProfile
import pstats
from django.core.management.commands.runserver import Command as RunServerCommand

class Command(RunServerCommand):
    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument('--profile', action='store_true', help='Enable profiling')
        parser.add_argument('--sort', default='cumulative', help='Sorting criteria for profiling output (cumulative, time, calls)')
        parser.add_argument('--restrict', nargs='*', help='Restrict profiling to specific functions or modules')
        parser.add_argument('--output', default='profile_output.prof', help='Output file for profiling data')

    def run(self, *args, **kwargs):
        if kwargs['profile']:
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                super().run(*args, **kwargs)
            finally:
                profiler.disable()
                with open(kwargs['output'], 'w') as f:
                    stats = pstats.Stats(profiler, stream=f)
                    if kwargs['restrict']:
                        stats.print_stats(*kwargs['restrict'])
                    else:
                        stats.strip_dirs().sort_stats(kwargs['sort']).print_stats()
                print(f"Profile data saved to {kwargs['output']}")
        else:
            super().run(*args, **kwargs)  # Run without profiling
