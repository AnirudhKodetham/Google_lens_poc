package com.programminghut.object_detection

import android.content.Intent
import android.graphics.*
import android.media.ExifInterface
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import androidx.core.content.FileProvider
import com.programminghut.object_detection.ml.SsdMobilenetV11Metadata1
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity() {

    val paint = Paint()
    lateinit var btnGallery: Button
    lateinit var btnCamera: Button
    lateinit var imageView: ImageView
    lateinit var bitmap: Bitmap
    lateinit var currentPhotoPath: String
    var colors = listOf<Int>(Color.BLUE, Color.GREEN, Color.RED, Color.CYAN, Color.GRAY, Color.BLACK,
        Color.DKGRAY, Color.MAGENTA, Color.YELLOW, Color.RED)
    lateinit var labels: List<String>
    lateinit var model: SsdMobilenetV11Metadata1
    val imageProcessor = ImageProcessor.Builder().add(ResizeOp(300, 300, ResizeOp.ResizeMethod.BILINEAR)).build()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        labels = FileUtil.loadLabels(this, "labels.txt")
        model = SsdMobilenetV11Metadata1.newInstance(this)

        paint.color = Color.BLUE
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 5.0f

        Log.d("labels", labels.toString())

        btnGallery = findViewById(R.id.btnGallery)
        btnCamera = findViewById(R.id.btnCamera)
        imageView = findViewById(R.id.imageView)

        // For selecting image from gallery
        btnGallery.setOnClickListener {
            val intent = Intent(Intent.ACTION_GET_CONTENT)
            intent.type = "image/*"
            startActivityForResult(intent, 101)
        }

        // For opening the camera
        btnCamera.setOnClickListener {
            dispatchTakePictureIntent()
        }
    }

    // Function to capture image from camera
    private fun dispatchTakePictureIntent() {
        val takePictureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        if (takePictureIntent.resolveActivity(packageManager) != null) {
            val photoFile: File? = try {
                createImageFile()
            } catch (ex: IOException) {
                ex.printStackTrace()
                null
            }
            photoFile?.also {
                val photoURI: Uri = FileProvider.getUriForFile(this, "com.programminghut.fileprovider", it)
                takePictureIntent.putExtra(MediaStore.EXTRA_OUTPUT, photoURI)
                startActivityForResult(takePictureIntent, 102)
            }
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss").format(Date())
        val storageDir: File = getExternalFilesDir(null)!!
        return File.createTempFile("JPEG_${timeStamp}_", ".jpg", storageDir).apply {
            currentPhotoPath = absolutePath
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 101 && data != null) {  // Gallery Image
            val uri = data.data
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            bitmap = fixImageOrientation(bitmap, uri.toString())  // Fix orientation
            get_predictions()
        } else if (requestCode == 102 && resultCode == RESULT_OK) {  // Camera Image
            bitmap = BitmapFactory.decodeFile(currentPhotoPath)
            bitmap = fixImageOrientation(bitmap, currentPhotoPath)  // Fix orientation
            get_predictions()
        }
    }

    // Fix the orientation of the image if needed
    fun fixImageOrientation(bitmap: Bitmap, imagePath: String): Bitmap {
        val exif: ExifInterface
        try {
            exif = ExifInterface(imagePath)
            val orientation = exif.getAttributeInt(ExifInterface.TAG_ORIENTATION, ExifInterface.ORIENTATION_UNDEFINED)
            val matrix = Matrix()
            when (orientation) {
                ExifInterface.ORIENTATION_ROTATE_90 -> matrix.postRotate(90F)
                ExifInterface.ORIENTATION_ROTATE_180 -> matrix.postRotate(180F)
                ExifInterface.ORIENTATION_ROTATE_270 -> matrix.postRotate(270F)
                else -> return bitmap
            }
            return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
        } catch (e: IOException) {
            e.printStackTrace()
        }
        return bitmap
    }

    override fun onDestroy() {
        super.onDestroy()
        model.close()
    }



    fun get_predictions() {
        var image = TensorImage.fromBitmap(bitmap)
        image = imageProcessor.process(image)
        val outputs = model.process(image)
        val locations = outputs.locationsAsTensorBuffer.floatArray
        val classes = outputs.classesAsTensorBuffer.floatArray
        val scores = outputs.scoresAsTensorBuffer.floatArray

        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(mutable)
        val h = mutable.height
        val w = mutable.width

        paint.textSize = h / 15f
        paint.strokeWidth = h / 85f

        // Find the highest score
        var maxScoreIndex = -1
        var maxScore = 0f

        scores.forEachIndexed { index, score ->
            if (score > maxScore) { // Update maxScore and its index if a higher score is found
                maxScore = score
                maxScoreIndex = index
            }
        }

        // If we found a valid detection with a score higher than 0.5, display it
        if (maxScoreIndex != -1 && maxScore > 0.5) {
            val label = labels[classes[maxScoreIndex].toInt()] + ": " + String.format("%.2f", maxScore * 100) + "%"

            val item = labels[classes[maxScoreIndex].toInt()]
            Log.d("Detected Object", item)


            paint.color = colors[maxScoreIndex]
            paint.style = Paint.Style.FILL
            canvas.drawText(label, 20f, h / 2f, paint)  // Display the highest prediction label
            // Log the detected object and its score
            Log.d("Detected Object", label)
        }

        imageView.setImageBitmap(mutable)
    }

//    fun get_predictions() {
//        var image = TensorImage.fromBitmap(bitmap)
//        image = imageProcessor.process(image)
//        val outputs = model.process(image)
//        val locations = outputs.locationsAsTensorBuffer.floatArray
//        val classes = outputs.classesAsTensorBuffer.floatArray
//        val scores = outputs.scoresAsTensorBuffer.floatArray
//
//        val mutable = bitmap.copy(Bitmap.Config.ARGB_8888, true)
//        val canvas = Canvas(mutable)
//        val h = mutable.height
//        val w = mutable.width
//
//        paint.textSize = h / 15f
//        paint.strokeWidth = h / 85f
//
//        scores.forEachIndexed { index, score ->
//            if (score > 0.5) {
//                var x = index * 4
//                paint.color = colors[index]
//                paint.style = Paint.Style.STROKE
//                canvas.drawRect(RectF(locations[x + 1] * w, locations[x] * h, locations[x + 3] * w, locations[x + 2] * h), paint)
//                paint.style = Paint.Style.FILL
//                val label = labels[classes[index].toInt()] + " " + score.toString()
//                canvas.drawText(label, locations[x + 1] * w, locations[x] * h, paint)
//
//                // Print the detected label in log
//                Log.d("Detected Object", label)
//            }
//        }
//
//        imageView.setImageBitmap(mutable)
//    }


}
