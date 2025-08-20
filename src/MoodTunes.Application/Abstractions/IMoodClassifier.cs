namespace MoodTunes.Application.Abstractions;

public interface IMoodClassifier
{
    Task<Dictionary<string,double>> ClassifyTextAsync(string text, CancellationToken ct = default);
}
