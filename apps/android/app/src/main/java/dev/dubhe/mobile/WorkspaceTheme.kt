package dev.dubhe.mobile

import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.Surface
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.scale
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

private val lightPalette = lightColorScheme(
    primary = Color(0xFF344F41), onPrimary = Color.White,
    primaryContainer = Color(0xFFDDE9E0), onPrimaryContainer = Color(0xFF22372C),
    secondary = Color(0xFF4D6357), onSecondary = Color.White,
    secondaryContainer = Color(0xFFE7EEE8), onSecondaryContainer = Color(0xFF30483B),
    background = Color(0xFFF7F6F0), onBackground = Color(0xFF1C2420),
    surface = Color(0xFFFDFDF9), onSurface = Color(0xFF1C2420),
    surfaceVariant = Color(0xFFE8EDE6), onSurfaceVariant = Color(0xFF4D5D54),
    surfaceContainerHigh = Color(0xFFEDF1EB), surfaceContainerHighest = Color(0xFFE2EAE2),
    outline = Color(0xFF77867B), outlineVariant = Color(0xFFD4DFD5),
    error = Color(0xFF9D3D2B), errorContainer = Color(0xFFFFE4DB), onErrorContainer = Color(0xFF48160B),
)

private val darkPalette = darkColorScheme(
    primary = Color(0xFFACCDBA), onPrimary = Color(0xFF20372B),
    primaryContainer = Color(0xFF344D40), onPrimaryContainer = Color(0xFFD9EDE0),
    secondary = Color(0xFFADC0B3), onSecondary = Color(0xFF26392E),
    secondaryContainer = Color(0xFF2F4238), onSecondaryContainer = Color(0xFFDBE8DF),
    background = Color(0xFF171D19), onBackground = Color(0xFFEDF1EC),
    surface = Color(0xFF212B24), onSurface = Color(0xFFEDF1EC),
    surfaceVariant = Color(0xFF2C3A31), onSurfaceVariant = Color(0xFFC0CEC3),
    surfaceContainerHigh = Color(0xFF2A382E), surfaceContainerHighest = Color(0xFF324239),
    outline = Color(0xFF899C8E), outlineVariant = Color(0xFF3C4B41),
    error = Color(0xFFF5AD99), errorContainer = Color(0xFF552D23), onErrorContainer = Color(0xFFFFDCD2),
)

private val defaults = Typography()
private val workspaceType = defaults.copy(
    headlineMedium = defaults.headlineMedium.copy(fontSize = 29.sp, fontWeight = FontWeight.SemiBold),
    headlineSmall = defaults.headlineSmall.copy(fontSize = 23.sp, fontWeight = FontWeight.SemiBold),
    titleMedium = defaults.titleMedium.copy(fontWeight = FontWeight.SemiBold, letterSpacing = 0.sp),
    bodyLarge = defaults.bodyLarge.copy(letterSpacing = 0.sp),
    bodyMedium = defaults.bodyMedium.copy(letterSpacing = 0.sp),
    labelLarge = defaults.labelLarge.copy(fontWeight = FontWeight.SemiBold, letterSpacing = 0.sp),
)

@Composable
internal fun DubheTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = if (isSystemInDarkTheme()) darkPalette else lightPalette,
        typography = workspaceType,
        shapes = Shapes(
            extraSmall = RoundedCornerShape(8.dp),
            small = RoundedCornerShape(12.dp),
            medium = RoundedCornerShape(16.dp),
            large = RoundedCornerShape(21.dp),
            extraLarge = RoundedCornerShape(26.dp),
        ),
        content = content,
    )
}

@Composable
internal fun WorkspaceCard(modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    Surface(
        modifier = modifier,
        shape = MaterialTheme.shapes.large,
        color = MaterialTheme.colorScheme.surface,
        border = BorderStroke(1.dp, MaterialTheme.colorScheme.outlineVariant),
        content = content,
    )
}

@Composable
internal fun WorkspaceBackdrop(modifier: Modifier) {
    val colors = MaterialTheme.colorScheme
    Canvas(modifier) {
        drawRect(Brush.verticalGradient(listOf(colors.primaryContainer, colors.background), endY = size.height * .62f))
    }
}

@Composable
internal fun DubheMark(size: Dp) {
    Image(
        painterResource(R.mipmap.ic_launcher), contentDescription = null,
        modifier = Modifier.size(size).clip(RoundedCornerShape(size * .28f)),
    )
}

internal enum class Glyph {
    Chat, Translate, Model, Settings, History, Add, Chevron, Send, Stop,
    Attachment, Close, Image, File, Copy, Download, Swap,
}

@Composable
internal fun IconAction(glyph: Glyph, label: String, enabled: Boolean = true, onClick: () -> Unit) {
    IconButton(onClick = onClick, enabled = enabled, modifier = Modifier.semantics { contentDescription = label }) {
        WorkspaceIcon(
            glyph,
            color = MaterialTheme.colorScheme.onSurfaceVariant.copy(alpha = if (enabled) 1f else .38f),
        )
    }
}

@Composable
internal fun WorkspaceIcon(
    glyph: Glyph,
    modifier: Modifier = Modifier,
    color: Color = MaterialTheme.colorScheme.onSurfaceVariant,
) {
    Canvas(modifier.size(22.dp)) {
        scale(size.width / 24f, size.height / 24f, pivot = Offset.Zero) {
            val stroke = Stroke(width = 1.65f, cap = StrokeCap.Round, join = StrokeJoin.Round)
            fun path(draw: Path.() -> Unit) = drawPath(Path().apply(draw), color, style = stroke)
            fun line(x: Float, y: Float, endX: Float, endY: Float) =
                drawLine(color, Offset(x, y), Offset(endX, endY), strokeWidth = 1.65f, cap = StrokeCap.Round)
            fun box(left: Float, top: Float, width: Float, height: Float) =
                drawRoundRect(color, Offset(left, top), Size(width, height), androidx.compose.ui.geometry.CornerRadius(2f), style = stroke)
            when (glyph) {
                Glyph.Chat -> {
                    path {
                        moveTo(3f, 10f)
                        quadraticTo(3f, 4f, 9f, 4f)
                        lineTo(15f, 4f)
                        quadraticTo(21f, 4f, 21f, 10f)
                        lineTo(21f, 12f)
                        quadraticTo(21f, 17.5f, 15f, 17.5f)
                        lineTo(10f, 17.5f)
                        lineTo(5f, 21f)
                        lineTo(5f, 16f)
                        quadraticTo(3f, 14f, 3f, 12f)
                        close()
                    }
                    for (x in listOf(8f, 12f, 16f)) drawCircle(color, .85f, Offset(x, 10.8f))
                }
                Glyph.Translate -> {
                    drawCircle(color, 8.5f, Offset(12f, 12f), style = stroke)
                    drawOval(color, Offset(8.5f, 3.5f), Size(7f, 17f), style = stroke)
                    line(4f, 12f, 20f, 12f)
                }
                Glyph.Model -> path {
                    moveTo(12f, 2.8f)
                    lineTo(20.8f, 7.8f)
                    lineTo(20.8f, 16.5f)
                    lineTo(12f, 21.3f)
                    lineTo(3.2f, 16.5f)
                    lineTo(3.2f, 7.8f)
                    close()
                    moveTo(3.2f, 7.8f)
                    lineTo(12f, 12.8f)
                    lineTo(20.8f, 7.8f)
                    moveTo(12f, 12.8f)
                    lineTo(12f, 21.3f)
                }
                Glyph.Settings -> {
                    for ((y, x) in listOf(6f to 8f, 12f to 15f, 18f to 11f)) {
                        line(4f, y, 20f, y)
                        drawCircle(color, 2.45f, Offset(x, y))
                    }
                }
                Glyph.History -> {
                    drawArc(color, 194f, 316f, false, Offset(4f, 4f), Size(16f, 16f), style = stroke)
                    path { moveTo(3.5f, 4.5f); lineTo(3.5f, 9f); lineTo(8f, 9f) }
                    path { moveTo(12f, 7.5f); lineTo(12f, 12.5f); lineTo(15.5f, 14.5f) }
                }
                Glyph.Add -> {
                    line(6f, 12f, 18f, 12f)
                    line(12f, 6f, 12f, 18f)
                }
                Glyph.Chevron -> path { moveTo(7f, 10f); lineTo(12f, 15f); lineTo(17f, 10f) }
                Glyph.Send -> {
                    line(12f, 19f, 12f, 5f)
                    path { moveTo(6.5f, 10.5f); lineTo(12f, 5f); lineTo(17.5f, 10.5f) }
                }
                Glyph.Stop -> drawRoundRect(
                    color, Offset(7f, 7f), Size(10f, 10f), androidx.compose.ui.geometry.CornerRadius(2f),
                )
                Glyph.Attachment -> path {
                    moveTo(8f, 14.5f)
                    lineTo(14.5f, 8f)
                    cubicTo(17f, 5.5f, 19.8f, 8.5f, 17.5f, 10.8f)
                    lineTo(10f, 18.5f)
                    cubicTo(6f, 22.5f, 1.5f, 17.5f, 5.3f, 13.7f)
                    lineTo(13.4f, 5.5f)
                    cubicTo(18f, .9f, 23.3f, 6.4f, 19.2f, 10.6f)
                }
                Glyph.Close -> {
                    line(7f, 7f, 17f, 17f)
                    line(17f, 7f, 7f, 17f)
                }
                Glyph.Image -> {
                    box(3.5f, 4f, 17f, 16f)
                    drawCircle(color, 1.5f, Offset(8.6f, 9f))
                    path { moveTo(4.2f, 17f); lineTo(10f, 12f); lineTo(13f, 14.5f); lineTo(16.5f, 11.5f); lineTo(20f, 15f) }
                }
                Glyph.File -> path {
                    moveTo(6f, 3.5f)
                    lineTo(14f, 3.5f)
                    lineTo(18.5f, 8f)
                    lineTo(18.5f, 20.5f)
                    lineTo(6f, 20.5f)
                    close()
                    moveTo(14f, 3.5f)
                    lineTo(14f, 8f)
                    lineTo(18.5f, 8f)
                }
                Glyph.Copy -> {
                    box(8f, 8f, 12f, 12f)
                    path { moveTo(15.5f, 4f); lineTo(4f, 4f); lineTo(4f, 15.5f) }
                }
                Glyph.Download -> {
                    line(12f, 4f, 12f, 15f)
                    path { moveTo(8f, 11f); lineTo(12f, 15f); lineTo(16f, 11f) }
                    path { moveTo(5f, 16f); lineTo(5f, 20f); lineTo(19f, 20f); lineTo(19f, 16f) }
                }
                Glyph.Swap -> {
                    path { moveTo(4f, 8f); lineTo(20f, 8f); lineTo(16f, 4f) }
                    path { moveTo(20f, 16f); lineTo(4f, 16f); lineTo(8f, 20f) }
                }
            }
        }
    }
}
