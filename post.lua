wrk.method = "POST"
wrk.headers["Content-Type"] = "application/json"

-- Simple JSON payload to send with each request
wrk.body = [[
{
  "instances": [
    [5.1, 3.5, 1.4, 0.2],
    [6.2, 2.8, 4.8, 1.8],
    [5.9, 3.0, 5.1, 1.8]
  ]
}
]]
