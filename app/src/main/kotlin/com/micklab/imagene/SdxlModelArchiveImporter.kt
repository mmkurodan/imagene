package com.micklab.imagene

import android.content.Context
import android.net.Uri
import java.io.BufferedInputStream
import java.io.File
import java.io.IOException
import java.io.InputStream
import java.util.zip.GZIPInputStream

data class ModelImportResult(
    val extractedFiles: Int,
    val destinationDir: File
)

object SdxlModelArchiveImporter {

    private const val TAG = "SdxlModelArchiveImporter"
    private const val BUFFER_SIZE = 64 * 1024
    private const val TAR_BLOCK_SIZE = 512

    fun importFromTarGz(context: Context, archiveUri: Uri): ModelImportResult {
        SdxlModelLoader.initialize(context)

        val targetDir = SdxlModelLoader.getModelBaseDir()
        val parentDir = targetDir.parentFile
            ?: throw IOException("Model directory parent is unavailable: ${targetDir.absolutePath}")
        if (!parentDir.exists() && !parentDir.mkdirs()) {
            throw IOException("Failed to create model storage root: ${parentDir.absolutePath}")
        }

        val pathPrefix = determineArchivePrefix(context, archiveUri)
        val stagingDir = File(parentDir, "${targetDir.name}_staging_${System.currentTimeMillis()}")
        if (stagingDir.exists() || !stagingDir.mkdirs()) {
            throw IOException("Failed to create staging directory: ${stagingDir.absolutePath}")
        }

        var backupDir: File? = null
        try {
            val extractedFiles = extractArchive(context, archiveUri, pathPrefix, stagingDir)
            if (extractedFiles == 0) {
                throw IllegalStateException("tar.gz に展開できるファイルがありません。")
            }

            val missingRequiredFiles = SdxlModelLoader.getRequiredRuntimeFiles()
                .filterNot { File(stagingDir, it).isFile }
            if (missingRequiredFiles.isNotEmpty()) {
                throw IllegalStateException(
                    "tar.gz に必要なファイルが含まれていません: ${missingRequiredFiles.joinToString(", ")}"
                )
            }

            backupDir = backupExistingDirectory(targetDir)
            moveDirectory(stagingDir, targetDir)
            backupDir?.deleteRecursively()

            AppLogStore.i(
                TAG,
                "Imported model archive to ${targetDir.absolutePath} (prefix=${pathPrefix.joinToString("/")}, files=$extractedFiles)"
            )
            return ModelImportResult(extractedFiles = extractedFiles, destinationDir = targetDir)
        } catch (e: Exception) {
            AppLogStore.e(TAG, "Failed to import model tar.gz from $archiveUri", e)
            restoreBackup(targetDir, backupDir)
            throw e
        } finally {
            if (stagingDir.exists()) {
                stagingDir.deleteRecursively()
            }
            if (backupDir?.exists() == true && targetDir.exists()) {
                backupDir.deleteRecursively()
            }
        }
    }

    fun resetModelDirectory(context: Context): File {
        SdxlModelLoader.initialize(context)
        val targetDir = SdxlModelLoader.getModelBaseDir()
        if (targetDir.exists() && !targetDir.deleteRecursively()) {
            throw IOException("Failed to clear model directory: ${targetDir.absolutePath}")
        }
        AppLogStore.i(TAG, "Reset model directory: ${targetDir.absolutePath}")
        return targetDir
    }

    private fun determineArchivePrefix(context: Context, archiveUri: Uri): List<String> {
        val entrySegments = mutableListOf<List<String>>()
        openTarGzInputStream(context, archiveUri).use { tarInput ->
            iterateTarEntries(tarInput) { entry, _ ->
                if (entry.isRegularFile) {
                    entrySegments += normalizeEntrySegments(entry.name)
                }
            }
        }

        if (entrySegments.isEmpty()) {
            return emptyList()
        }

        val requiredPaths = SdxlModelLoader.getRequiredRuntimeFiles()
            .map { normalizeEntrySegments(it) }
        for (requiredPath in requiredPaths) {
            val match = entrySegments.firstOrNull { it.size >= requiredPath.size && it.takeLast(requiredPath.size) == requiredPath }
            if (match != null) {
                return match.dropLast(requiredPath.size)
            }
        }

        val commonPrefix = entrySegments.drop(1).fold(entrySegments.first()) { prefix, segments ->
            commonPrefix(prefix, segments)
        }
        return if (commonPrefix.size == 1) commonPrefix else emptyList()
    }

    private fun extractArchive(
        context: Context,
        archiveUri: Uri,
        pathPrefix: List<String>,
        stagingDir: File
    ): Int {
        var extractedFiles = 0

        openTarGzInputStream(context, archiveUri).use { tarInput ->
            iterateTarEntries(tarInput) { entry, entryInput ->
                if (!entry.isDirectory && !entry.isRegularFile) {
                    return@iterateTarEntries
                }

                val entrySegments = normalizeEntrySegments(entry.name)
                val relativeSegments = stripPrefix(entrySegments, pathPrefix)

                if (relativeSegments != null && relativeSegments.isNotEmpty()) {
                    val destination = resolveDestination(stagingDir, relativeSegments)
                    if (entry.isDirectory) {
                        if (!destination.exists() && !destination.mkdirs()) {
                            throw IOException("Failed to create directory: ${destination.absolutePath}")
                        }
                    } else {
                        val parent = destination.parentFile
                        if (parent != null && !parent.exists() && !parent.mkdirs()) {
                            throw IOException("Failed to create directory: ${parent.absolutePath}")
                        }
                        destination.outputStream().buffered().use { output ->
                            entryInput.copyTo(output, BUFFER_SIZE)
                        }
                        extractedFiles += 1
                    }
                }
            }
        }

        return extractedFiles
    }

    private data class TarEntry(
        val name: String,
        val size: Long,
        val typeFlag: Char
    ) {
        val isDirectory: Boolean
            get() = typeFlag == '5' || name.endsWith("/")

        val isRegularFile: Boolean
            get() = typeFlag == '0' || typeFlag == '\u0000'
    }

    private class TarEntryInputStream(
        private val source: InputStream,
        private val size: Long
    ) : InputStream() {
        private var consumed: Long = 0L

        override fun read(): Int {
            if (consumed >= size) return -1
            val value = source.read()
            if (value == -1) {
                throw IOException("tar.gz が途中で終了しています。")
            }
            consumed += 1
            return value
        }

        override fun read(buffer: ByteArray, offset: Int, length: Int): Int {
            if (consumed >= size) return -1
            val toRead = minOf(length.toLong(), size - consumed).toInt()
            val count = source.read(buffer, offset, toRead)
            if (count == -1) {
                throw IOException("tar.gz が途中で終了しています。")
            }
            consumed += count.toLong()
            return count
        }

        fun skipRemaining() {
            var remaining = size - consumed
            while (remaining > 0) {
                val skipped = source.skip(remaining)
                if (skipped > 0) {
                    remaining -= skipped
                } else {
                    val value = source.read()
                    if (value == -1) {
                        throw IOException("tar.gz が途中で終了しています。")
                    }
                    remaining -= 1
                }
            }
            consumed = size
        }
    }

    private fun iterateTarEntries(
        tarInput: InputStream,
        onEntry: (TarEntry, TarEntryInputStream) -> Unit
    ) {
        val header = ByteArray(TAR_BLOCK_SIZE)
        while (readBlockOrEof(tarInput, header)) {
            if (header.all { it.toInt() == 0 }) {
                return
            }

            val entry = parseTarEntry(header)
            val entryInput = TarEntryInputStream(tarInput, entry.size)
            onEntry(entry, entryInput)
            entryInput.skipRemaining()

            val padding = ((TAR_BLOCK_SIZE - (entry.size % TAR_BLOCK_SIZE)) % TAR_BLOCK_SIZE).toInt()
            if (padding > 0) {
                skipFully(tarInput, padding.toLong())
            }
        }
    }

    private fun readBlockOrEof(input: InputStream, buffer: ByteArray): Boolean {
        var offset = 0
        while (offset < buffer.size) {
            val read = input.read(buffer, offset, buffer.size - offset)
            if (read == -1) {
                if (offset == 0) {
                    return false
                }
                throw IOException("tar.gz ヘッダーが途中で終了しています。")
            }
            offset += read
        }
        return true
    }

    private fun parseTarEntry(header: ByteArray): TarEntry {
        val name = readTarString(header, 0, 100)
        val prefix = readTarString(header, 345, 155)
        val fullName = when {
            prefix.isNotBlank() && name.isNotBlank() -> "$prefix/$name"
            prefix.isNotBlank() -> prefix
            else -> name
        }
        if (fullName.isBlank()) {
            throw IOException("tar.gz 内に不正なエントリ名が含まれています。")
        }

        val size = parseTarSize(header, 124, 12)
        val typeRaw = header[156].toInt() and 0xff
        val typeFlag = if (typeRaw == 0) '\u0000' else typeRaw.toChar()
        return TarEntry(name = fullName, size = size, typeFlag = typeFlag)
    }

    private fun readTarString(header: ByteArray, offset: Int, length: Int): String {
        val end = (offset until (offset + length))
            .firstOrNull { header[it].toInt() == 0 } ?: (offset + length)
        return String(header, offset, end - offset, Charsets.US_ASCII).trim()
    }

    private fun parseTarSize(header: ByteArray, offset: Int, length: Int): Long {
        val value = readTarString(header, offset, length)
        if (value.isBlank()) {
            return 0L
        }
        return value.toLongOrNull(radix = 8)
            ?: throw IOException("tar.gz 内に不正なサイズが含まれています: $value")
    }

    private fun backupExistingDirectory(targetDir: File): File? {
        if (!targetDir.exists()) {
            return null
        }

        val backupDir = File(
            targetDir.parentFile,
            "${targetDir.name}_backup_${System.currentTimeMillis()}"
        )
        moveDirectory(targetDir, backupDir)
        return backupDir
    }

    private fun restoreBackup(targetDir: File, backupDir: File?) {
        if (backupDir == null || !backupDir.exists()) {
            return
        }

        if (targetDir.exists()) {
            targetDir.deleteRecursively()
        }
        moveDirectory(backupDir, targetDir)
    }

    private fun moveDirectory(sourceDir: File, targetDir: File) {
        if (!sourceDir.exists()) {
            throw IOException("Source directory does not exist: ${sourceDir.absolutePath}")
        }

        val targetParent = targetDir.parentFile
        if (targetParent != null && !targetParent.exists() && !targetParent.mkdirs()) {
            throw IOException("Failed to create directory: ${targetParent.absolutePath}")
        }

        if (targetDir.exists() && !targetDir.deleteRecursively()) {
            throw IOException("Failed to clear directory: ${targetDir.absolutePath}")
        }

        if (sourceDir.renameTo(targetDir)) {
            return
        }

        if (!sourceDir.copyRecursively(targetDir, overwrite = true)) {
            throw IOException("Failed to copy directory to ${targetDir.absolutePath}")
        }
        if (!sourceDir.deleteRecursively()) {
            throw IOException("Failed to delete temporary directory: ${sourceDir.absolutePath}")
        }
    }

    private fun openTarGzInputStream(context: Context, archiveUri: Uri): InputStream {
        val inputStream = context.contentResolver.openInputStream(archiveUri)
            ?: throw IOException("tar.gz ファイルを開けませんでした。")
        return GZIPInputStream(BufferedInputStream(inputStream), BUFFER_SIZE)
    }

    private fun normalizeEntrySegments(entryName: String): List<String> {
        val normalized = entryName.replace('\\', '/')
        return normalized.split('/')
            .filter { it.isNotBlank() && it != "." }
            .also { segments ->
                if (segments.any { it == ".." }) {
                    throw IOException("tar.gz 内に不正なパスが含まれています: $entryName")
                }
            }
    }

    private fun stripPrefix(segments: List<String>, prefix: List<String>): List<String>? {
        if (prefix.isEmpty()) {
            return segments
        }
        if (segments.size < prefix.size || segments.subList(0, prefix.size) != prefix) {
            return null
        }
        return segments.drop(prefix.size)
    }

    private fun resolveDestination(baseDir: File, relativeSegments: List<String>): File {
        val destination = relativeSegments.fold(baseDir) { current, segment ->
            File(current, segment)
        }
        val canonicalBase = baseDir.canonicalFile
        val canonicalDestination = destination.canonicalFile
        val allowedPrefix = canonicalBase.path + File.separator
        if (canonicalDestination.path != canonicalBase.path &&
            !canonicalDestination.path.startsWith(allowedPrefix)
        ) {
            throw IOException("tar.gz 内に不正な展開先が含まれています: ${destination.path}")
        }
        return canonicalDestination
    }

    private fun commonPrefix(left: List<String>, right: List<String>): List<String> {
        val limit = minOf(left.size, right.size)
        val prefix = ArrayList<String>(limit)
        for (index in 0 until limit) {
            if (left[index] != right[index]) {
                break
            }
            prefix += left[index]
        }
        return prefix
    }

    private fun skipFully(input: InputStream, byteCount: Long) {
        var remaining = byteCount
        while (remaining > 0) {
            val skipped = input.skip(remaining)
            if (skipped > 0) {
                remaining -= skipped
            } else {
                val value = input.read()
                if (value == -1) {
                    throw IOException("tar.gz が途中で終了しています。")
                }
                remaining -= 1
            }
        }
    }
}
