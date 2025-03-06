import httpx
from clerk_backend_api import Clerk
from clerk_backend_api.jwks_helpers import AuthenticateRequestOptions
from decouple import config
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

CLERK_SECRET_KEY = config("CLERK_SECRET_KEY")
ENVIRONMENT = config("ENVIRONMENT", default="production")
DOMAIN = config("DOMAIN")
CLERK_DOMAIN = config("CLERK_DOMAIN")
APP_URL_VERCEL = config("APP_URL_VERCEL")

app = FastAPI()

# CORS settings (adjust as needed)
allowed_origins = [
    f"https://{CLERK_DOMAIN}",
    f"https://{DOMAIN}",
    f"https://{APP_URL_VERCEL}",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def clerk_auth_middleware(request: Request, call_next):
    # Allow preflight OPTIONS
    if request.method.lower() == "options":
        return await call_next(request)

    # Paths that do NOT require authentication
    unprotected_paths = ["/login.html", "/favicon.ico"]
    if any(request.url.path.startswith(path) for path in unprotected_paths):
        return await call_next(request)

    # Instantiate the Clerk SDK
    clerk = Clerk(bearer_auth=CLERK_SECRET_KEY)

    # Convert FastAPI request -> httpx.Request
    client_request = httpx.Request(
        method=request.method,
        url=str(request.url),
        headers=dict(request.headers)
    )

    # Set up authorized parties depending on environment
    if ENVIRONMENT == "development":
        authorized_parties = [
            f"https://{CLERK_DOMAIN}",
            f"https://{DOMAIN}",
            f"https://{APP_URL_VERCEL}",
            "http://0.0.0.0:8000",
            "http://localhost:8000",
        ]
    else:
        authorized_parties = [
            f"https://{CLERK_DOMAIN}",
            f"https://{DOMAIN}",
            f"https://{APP_URL_VERCEL}",
        ]

    options = AuthenticateRequestOptions(authorized_parties=authorized_parties)

    try:
        # Validate token with Clerk
        auth_state = clerk.authenticate_request(client_request, options)
    except Exception:
        # Any token/validation error => redirect to login
        return RedirectResponse(url="/login.html")

    # If user not signed in => redirect to login
    if not auth_state.is_signed_in:
        return RedirectResponse(url="/login.html")

    # If everything is okay, proceed
    response = await call_next(request)
    return response

# Mount your Sphinx docs as static files at "/"
app.mount("/", StaticFiles(directory="./main", html=True), name="main")
