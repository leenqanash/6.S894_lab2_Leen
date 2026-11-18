module MandelbrotCore
export coordinate_x, coordinate_y, reference, DEFAULT_SIZE, DEFAULT_ITERS
const DEFAULT_SIZE = 8192
const DEFAULT_ITERS = 2000
const ZOOM = Float32(1 / 10000)
const WX = Float32(-0.743643887 - 0.5 * Float64(ZOOM))
const WY = Float32(0.131825904 - 0.5 * Float64(ZOOM))
@inline coordinate_x(x, n) = (Float32(x - 1) / Float32(n)) * ZOOM + WX
@inline coordinate_y(y, n) = (Float32(y - 1) / Float32(n)) * ZOOM + WY

# Independent scalar CPU oracle. Output[y,x] uses Julia column-major storage.
function reference(n, limit=DEFAULT_ITERS)
    out = zeros(UInt32, n, n)
    for x in 1:n, y in 1:n
        cx = coordinate_x(x,n); cy = coordinate_y(y,n)
        x2 = 0f0; y2 = 0f0; w = 0f0; count = UInt32(0)
        while x2 + y2 <= 4f0 && count < limit
            real = x2 - y2 + cx
            imag = w - (x2 + y2) + cy
            x2 = real * real; y2 = imag * imag
            z = real + imag; w = z * z
            count += UInt32(1)
        end
        out[y,x] = count
    end
    out
end
end
