using Microsoft.AspNetCore.Mvc;
using MoodTunes.Application.Abstractions;

namespace MoodTunes.Api.Controllers;

[ApiController]
[Route("api/v1/mood")]
public class MoodController : ControllerBase
{
    private readonly IMoodClassifier _clf;
    public MoodController(IMoodClassifier clf) { _clf = clf; }

    public record TextReq(string Text);

    [HttpPost("analyze")]
    public async Task<IActionResult> Analyze([FromBody] TextReq req, CancellationToken ct)
    {
        if (string.IsNullOrWhiteSpace(req.Text))
            return BadRequest(new { error = "text is required" });

        var scores = await _clf.ClassifyTextAsync(req.Text, ct);
        var top = scores.OrderByDescending(kv => kv.Value).First();
        return Ok(new { labels = scores, top = new { mood = top.Key, score = top.Value } });
    }
}
