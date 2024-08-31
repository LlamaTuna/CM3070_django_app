import cProfile
import pstats
import io
from django.http import HttpResponse

class ProfilerMiddleware:
    """
    Middleware for profiling Django views.

    This middleware allows you to profile specific views by adding a 'profile' 
    parameter to the URL. When profiling is enabled, the middleware captures 
    performance statistics using cProfile and returns them as part of the HTTP 
    response.

    Attributes:
        get_response (callable): The next middleware or view in the Django 
        request/response cycle.
    """

    def __init__(self, get_response):
        """
        Initializes the middleware with the next callable in the Django 
        request/response cycle.

        Args:
            get_response (callable): The next middleware or view function in 
            the chain.
        """
        self.get_response = get_response

    def __call__(self, request):
        """
        Handles incoming requests and optionally profiles the request.

        If the 'profile' parameter is present in the request's GET data, the 
        request is profiled using cProfile, and the profiling results are 
        returned in the HTTP response. Otherwise, the request is processed 
        normally.

        Args:
            request (HttpRequest): The incoming HTTP request.

        Returns:
            HttpResponse: The HTTP response with or without profiling results.
        """
        if 'profile' in request.GET:
            # Enable the profiler
            profiler = cProfile.Profile()
            profiler.enable()

            # Process the request and get the response
            response = self.get_response(request)

            # Disable the profiler and collect the stats
            profiler.disable()
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s).sort_stats('cumtime')
            ps.print_stats()

            # Return the profiling results in the HTTP response
            return HttpResponse(f"<pre>{s.getvalue()}</pre>")
        else:
            # Process the request normally if 'profile' is not in the query parameters
            return self.get_response(request)
