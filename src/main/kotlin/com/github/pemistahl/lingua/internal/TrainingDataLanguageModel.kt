/*
 * Copyright © 2018-today Peter M. Stahl pemistahl@gmail.com
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either expressed or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.github.pemistahl.lingua.internal

import com.github.pemistahl.lingua.api.Language
import com.github.pemistahl.lingua.internal.util.extension.incrementCounter
import kotlinx.serialization.Serializable
import kotlinx.serialization.decodeFromString
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.util.TreeMap

@Serializable
internal data class JsonLanguageModel(val language: Language, val ngrams: Map<Fraction, String>)

internal data class TrainingDataLanguageModel(
    val language: Language,
    val absoluteFrequencies: Map<Ngram, Int>,
    val relativeFrequencies: Map<Ngram, Fraction>,
    val jsonRelativeFrequencies: RelativeFrequencies
) {
    fun getRelativeFrequency(ngram: Ngram): Float = jsonRelativeFrequencies[ngram.value]

    fun toJson(): String {
        val ngrams = mutableMapOf<Fraction, MutableList<Ngram>>()

        for ((ngram, fraction) in relativeFrequencies) {
            ngrams.computeIfAbsent(fraction) { mutableListOf() }.add(ngram)
        }

        val jsonLanguageModel = JsonLanguageModel(language, ngrams.mapValues { it.value.joinToString(separator = " ") })

        return Json.encodeToString(jsonLanguageModel)
    }

    companion object {
        fun fromText(
            text: Sequence<String>,
            language: Language,
            ngramLength: Int,
            charClass: String,
            lowerNgramAbsoluteFrequencies: Map<Ngram, Int>
        ): TrainingDataLanguageModel {

            require(ngramLength in 1..5) {
                "ngram length $ngramLength is not in range 1..5"
            }

            val absoluteFrequencies = computeAbsoluteFrequencies(
                text,
                ngramLength,
                charClass
            )

            val relativeFrequencies = computeRelativeFrequencies(
                ngramLength,
                absoluteFrequencies,
                lowerNgramAbsoluteFrequencies
            )

            return TrainingDataLanguageModel(
                language,
                absoluteFrequencies,
                relativeFrequencies,
                RelativeFrequencies.build(List(5) { TreeMap() })
            )
        }

        fun fromJson(json: String): TrainingDataLanguageModel {
            val jsonLanguageModel = Json.decodeFromString<JsonLanguageModel>(json)
            val jsonRelativeFrequencyMaps = List(5) { TreeMap<String, Float>() }

            for ((fraction, ngrams) in jsonLanguageModel.ngrams) {
                val fractionAsFloat = fraction.toFloat()
                for (ngram in ngrams.split(' ')) {
                    jsonRelativeFrequencyMaps[ngram.length - 1][ngram] = fractionAsFloat
                }
            }

            return TrainingDataLanguageModel(
                language = jsonLanguageModel.language,
                absoluteFrequencies = emptyMap(),
                relativeFrequencies = emptyMap(),
                jsonRelativeFrequencies = RelativeFrequencies.build(jsonRelativeFrequencyMaps)
            )
        }

        private fun computeAbsoluteFrequencies(
            text: Sequence<String>,
            ngramLength: Int,
            charClass: String
        ): Map<Ngram, Int> {

            val absoluteFrequencies = mutableMapOf<Ngram, Int>()
            val regex = Regex("[$charClass]+")

            for (line in text) {
                val lowerCasedLine = line.lowercase()
                for (i in 0..lowerCasedLine.length - ngramLength) {
                    val textSlice = lowerCasedLine.substring(i, i + ngramLength)
                    if (regex.matches(textSlice)) {
                        val ngram = Ngram(textSlice)
                        absoluteFrequencies.incrementCounter(ngram)
                    }
                }
            }

            return absoluteFrequencies
        }

        private fun computeRelativeFrequencies(
            ngramLength: Int,
            absoluteFrequencies: Map<Ngram, Int>,
            lowerNgramAbsoluteFrequencies: Map<Ngram, Int>
        ): Map<Ngram, Fraction> {

            val ngramProbabilities = mutableMapOf<Ngram, Fraction>()
            val totalNgramFrequency = absoluteFrequencies.values.sum()

            for ((ngram, frequency) in absoluteFrequencies) {
                val denominator = if (ngramLength == 1 || lowerNgramAbsoluteFrequencies.isEmpty()) {
                    totalNgramFrequency
                } else {
                    lowerNgramAbsoluteFrequencies.getValue(Ngram(ngram.value.substring(0, ngramLength - 1)))
                }
                ngramProbabilities[ngram] = Fraction(frequency, denominator)
            }

            return ngramProbabilities
        }
    }

    internal class RelativeFrequencies private constructor(private val data: Array<Entries>) {

        operator fun get(ngram: String): Float = data[ngram.length - 1][ngram]

        private class Entries(private val chars: CharArray, private val frequencies: FloatArray) {

            val size get() = frequencies.size

            operator fun get(ngram: String): Float {
                var low = 0
                var high = size - 1
                while (low <= high) {
                    val middle = (low + high) / 2
                    val diff = compareNgram(middle, ngram)
                    if (diff < 0) low = middle + 1
                    else if (diff > 0) high = middle - 1
                    else return frequencies[middle]
                }
                return 0F
            }

            private fun compareNgram(pos: Int, ngram: String): Int {
                val base = pos * ngram.length
                repeat(ngram.length) { i ->
                    val diff = chars[base + i].compareTo(ngram[i])
                    if (diff != 0) return diff
                }
                return 0
            }
        }

        companion object {
            fun build(jsonRelativeFrequencyMaps: List<TreeMap<String, Float>>): RelativeFrequencies {
                val data = jsonRelativeFrequencyMaps.map { map ->
                    val chars = map.keys.joinToString(separator = "").toCharArray()
                    val float = map.values.toFloatArray()
                    Entries(chars, float)
                }
                return RelativeFrequencies(data.toTypedArray())
            }
        }
    }
}
