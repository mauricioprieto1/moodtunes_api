using MoodTunes.Application.Abstractions;
using MoodTunes.Infrastructure.Classifiers;

var builder = WebApplication.CreateBuilder(args);

// Controllers & CORS
builder.Services.AddControllers();
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p
    .WithOrigins("http://localhost:5173", "http://localhost:3000")
    .AllowAnyHeader().AllowAnyMethod().AllowCredentials()
));

// DI: stub classifier for now
builder.Services.AddScoped<IMoodClassifier, LocalRuleBasedClassifier>();

// 🔹 Swagger registration (these two lines register ISwaggerProvider)
builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen();

var app = builder.Build();

// 🔹 Serve Swagger (dev only is fine)
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();    // serves /swagger/v1/swagger.json
    app.UseSwaggerUI();  // serves /swagger
}

// Pipeline
app.UseCors();
app.MapControllers();
app.MapGet("/_health", () => Results.Ok(new { ok = true }));
app.MapGet("/api/v1/ping", () => Results.Ok(new { ok = true, service = "moodtunes-api" }));

app.Run();
