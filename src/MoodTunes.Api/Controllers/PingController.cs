using Microsoft.AspNetCore.Mvc;

namespace MoodTunes.Api.Controllers;

[ApiController]
[Route("api/v1/ping2")]
public class PingController : ControllerBase
{
    [HttpGet]
    public IActionResult Get() => Ok(new { ok = true, ctrl = "ping2" });
}
