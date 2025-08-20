using MoodTunes.Application.Abstractions;
using System.Text.RegularExpressions;

namespace MoodTunes.Infrastructure.Classifiers;

public class LocalRuleBasedClassifier : IMoodClassifier
{
    private static readonly (string label, Regex rx, double score)[] rules = new[]
    {
        ("happy",       new Regex(@"\b(happy|joy|awesome|love it)\b", RegexOptions.IgnoreCase), 0.8),
        ("calm",        new Regex(@"\b(chill|calm|cozy|peaceful|lofi|coffee)\b", RegexOptions.IgnoreCase), 0.7),
        ("sad",         new Regex(@"\b(sad|down|upset)\b", RegexOptions.IgnoreCase), 0.8),
        ("melancholy",  new Regex(@"\b(nostalgia|nostalgic|miss|remember when|years ago)\b", RegexOptions.IgnoreCase), 0.7),
        ("energetic",   new Regex(@"\b(hyped|pumped|excited|let's go|workout)\b", RegexOptions.IgnoreCase), 0.75),
        ("angry",       new Regex(@"\b(angry|mad|furious|rage)\b", RegexOptions.IgnoreCase), 0.8),
        ("anxious",     new Regex(@"\b(anxious|nervous|stressed|worried)\b", RegexOptions.IgnoreCase), 0.75),
        ("romantic",    new Regex(@"\b(love|date night|romantic)\b", RegexOptions.IgnoreCase), 0.7),
    };

    public Task<Dictionary<string,double>> ClassifyTextAsync(string text, CancellationToken ct = default)
    {
        var scores = new Dictionary<string,double>(StringComparer.OrdinalIgnoreCase);
        foreach (var (label, rx, score) in rules)
            if (rx.IsMatch(text)) scores[label] = Math.Max(scores.GetValueOrDefault(label, 0), score);
        if (scores.Count == 0) scores["calm"] = 0.4; // fallback
        return Task.FromResult(scores);
    }
}
