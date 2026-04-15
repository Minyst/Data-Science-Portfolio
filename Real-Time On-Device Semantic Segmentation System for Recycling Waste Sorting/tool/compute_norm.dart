import 'dart:io';
import 'dart:math' as math;
import 'package:image/image.dart' as img;

// Usage: dart run tool/compute_norm.dart <folder>
// Recursively scans images and computes per-channel mean/std in RGB (0..1 range)
void main(List<String> args) async {
  if (args.isEmpty) {
    stderr.writeln('Usage: dart run tool/compute_norm.dart <folder>');
    exit(1);
  }
  final dir = Directory(args[0]);
  if (!await dir.exists()) {
    stderr.writeln('Folder not found: ${dir.path}');
    exit(2);
  }

  final exts = {'.jpg', '.jpeg', '.png', '.webp'};
  final files = <File>[];
  await for (final entity in dir.list(recursive: true, followLinks: false)) {
    if (entity is File) {
      final ext = entity.path.toLowerCase().split('.').last;
      if (exts.contains('.$ext')) files.add(entity);
    }
  }
  if (files.isEmpty) {
    stderr.writeln('No images found in ${dir.path}');
    exit(3);
  }

  double sumR = 0, sumG = 0, sumB = 0;
  double sqR = 0, sqG = 0, sqB = 0;
  int count = 0;

  for (final f in files) {
    try {
      final bytes = await f.readAsBytes();
      final im = img.decodeImage(bytes);
      if (im == null) continue;
      final w = im.width, h = im.height;
      for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
          final p = im.getPixel(x, y);
          final r = p.r / 255.0;
          final g = p.g / 255.0;
          final b = p.b / 255.0;
          sumR += r; sumG += g; sumB += b;
          sqR += r * r; sqG += g * g; sqB += b * b;
        }
      }
      count += w * h;
    } catch (_) {}
  }

  double meanR = sumR / count;
  double meanG = sumG / count;
  double meanB = sumB / count;
  double stdR = math.sqrt((sqR / count) - meanR * meanR).clamp(1e-6, 1e6);
  double stdG = math.sqrt((sqG / count) - meanG * meanG).clamp(1e-6, 1e6);
  double stdB = math.sqrt((sqB / count) - meanB * meanB).clamp(1e-6, 1e6);

  // Print with 6 decimals for easy copy-paste
  String f6(double v) => v.toStringAsFixed(6);
  stdout.writeln('Images: ${files.length}, Pixels: $count');
  stdout.writeln('meanR=${f6(meanR)}, meanG=${f6(meanG)}, meanB=${f6(meanB)}');
  stdout.writeln('stdR=${f6(stdR)}, stdG=${f6(stdG)}, stdB=${f6(stdB)}');
}


