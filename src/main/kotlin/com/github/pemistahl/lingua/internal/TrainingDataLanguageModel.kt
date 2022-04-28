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
import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import java.util.*

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
                RelativeFrequencies.build(emptySequence())
            )
        }

        fun fromJson(language: Language, jsonLanguageModels: Sequence<JsonLanguageModel>): TrainingDataLanguageModel {
            val jsonDataSequence: Sequence<Pair<String, Float>> = sequence {
                for (jsonLanguageModel in jsonLanguageModels) {
                    for ((fraction, ngrams) in jsonLanguageModel.ngrams) {
                        val fractionAsFloat = fraction.toFloat()
                        yieldAll(ngrams.splitToSequence(' ').map { it to fractionAsFloat })
                    }
                }
            }

            return TrainingDataLanguageModel(
                language = language,
                absoluteFrequencies = emptyMap(),
                relativeFrequencies = emptyMap(),
                jsonRelativeFrequencies = RelativeFrequencies.build(jsonDataSequence)
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

    /**
     * N-ary search tree.
     */
    internal sealed class RelativeFrequencies {

        abstract val frequency: Float

        operator fun get(ngram: CharSequence): Float = getImpl(ngram, 0)

        protected abstract fun getImpl(ngram: CharSequence, depth: Int): Float

        private class GenericNode(
            override val frequency: Float,
            private val keys: CharArray,
            private val values: Array<RelativeFrequencies>
        ) : RelativeFrequencies() {
            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                val key = ngram[depth]
                val i = if (keys.size <= 1024) {
                    keys.binarySearch(key)
                } else {
                    val first = ngram.first()
                    if (key < first) return 0F
                    if (key == first) return values.first().getImpl(ngram, depth + 1)
                    val last = ngram.last()
                    if (key > last) return 0F
                    if (key == last) return values.last().getImpl(ngram, depth + 1)
                    keys.binarySearch(key, 1, keys.size - 1)
                }
                return if (i >= 0) values[i].getImpl(ngram, depth + 1) else 0F
            }
        }

        /**
         * A node with one leaf child
         */
        private data class PreLeaf1(
            override val frequency: Float,
            private val key: Char,
            private val value: Float
        ) : RelativeFrequencies() {
            override fun getImpl(ngram: CharSequence, depth: Int) =
                when {
                    depth == ngram.length -> frequency
                    depth + 1 == ngram.length && key == ngram[depth] -> value
                    else -> 0F
                }
        }

        /**
         * A node with only 2 leaf children
         */
        private data class PreLeaf2(
            override val frequency: Float,
            private val key0: Char,
            private val key1: Char,
            private val value0: Float,
            private val value1: Float
        ) : RelativeFrequencies() {

            constructor(frequency: Float, keys: CharArray, values: FloatArray) :
                this(
                    frequency = frequency,
                    key0 = keys[0],
                    key1 = keys[1],
                    value0 = values[0],
                    value1 = values[1]
                )

            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                if (depth + 1 == ngram.length) {
                    val key = ngram[depth]
                    if (key0 == key) return value0
                    if (key1 == key) return value1
                }
                return 0F
            }
        }

        /**
         * A node with only 3 leaf children
         */
        private data class PreLeaf3(
            override val frequency: Float,
            private val key0: Char,
            private val key1: Char,
            private val key2: Char,
            private val value0: Float,
            private val value1: Float,
            private val value2: Float
        ) : RelativeFrequencies() {

            constructor(frequency: Float, keys: CharArray, values: FloatArray) :
                this(
                    frequency = frequency,
                    key0 = keys[0],
                    key1 = keys[1],
                    key2 = keys[2],
                    value0 = values[0],
                    value1 = values[1],
                    value2 = values[2]
                )

            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                if (depth + 1 == ngram.length) {
                    val key = ngram[depth]
                    if (key0 == key) return value0
                    if (key1 == key) return value1
                    if (key2 == key) return value2
                }
                return 0F
            }
        }

        /**
         * A node with only 4 leaf children
         */
        private data class PreLeaf4(
            override val frequency: Float,
            private val key0: Char,
            private val key1: Char,
            private val key2: Char,
            private val key3: Char,
            private val value0: Float,
            private val value1: Float,
            private val value2: Float,
            private val value3: Float
        ) : RelativeFrequencies() {

            constructor(frequency: Float, keys: CharArray, values: FloatArray) :
                this(
                    frequency = frequency,
                    key0 = keys[0],
                    key1 = keys[1],
                    key2 = keys[2],
                    key3 = keys[3],
                    value0 = values[0],
                    value1 = values[1],
                    value2 = values[2],
                    value3 = values[3]
                )

            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                if (depth + 1 == ngram.length) {
                    val key = ngram[depth]
                    if (key <= key0) {
                        if (key == key0) return value0
                    } else if (key >= key3) {
                        if (key == key3) return value3
                    } else {
                        if (key1 == key) return value1
                        if (key2 == key) return value2
                    }
                }
                return 0F
            }
        }

        /**
         * A node with only 5 leaf children
         */
        private data class PreLeaf5(
            override val frequency: Float,
            private val key0: Char,
            private val key1: Char,
            private val key2: Char,
            private val key3: Char,
            private val key4: Char,
            private val value0: Float,
            private val value1: Float,
            private val value2: Float,
            private val value3: Float,
            private val value4: Float
        ) : RelativeFrequencies() {

            constructor(frequency: Float, keys: CharArray, values: FloatArray) :
                this(
                    frequency = frequency,
                    key0 = keys[0],
                    key1 = keys[1],
                    key2 = keys[2],
                    key3 = keys[3],
                    key4 = keys[4],
                    value0 = values[0],
                    value1 = values[1],
                    value2 = values[2],
                    value3 = values[3],
                    value4 = values[4]
                )

            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                if (depth + 1 == ngram.length) {
                    val key = ngram[depth]
                    if (key <= key0) {
                        if (key == key0) return value0
                    } else if (key >= key4) {
                        if (key == key4) return value4
                    } else {
                        if (key1 == key) return value1
                        if (key2 == key) return value2
                        if (key3 == key) return value3
                    }
                }
                return 0F
            }
        }

        /**
         * A node with only 6 leaf children
         */
        private data class PreLeaf6(
            override val frequency: Float,
            private val key0: Char,
            private val key1: Char,
            private val key2: Char,
            private val key3: Char,
            private val key4: Char,
            private val key5: Char,
            private val value0: Float,
            private val value1: Float,
            private val value2: Float,
            private val value3: Float,
            private val value4: Float,
            private val value5: Float
        ) : RelativeFrequencies() {

            constructor(frequency: Float, keys: CharArray, values: FloatArray) :
                this(
                    frequency = frequency,
                    key0 = keys[0],
                    key1 = keys[1],
                    key2 = keys[2],
                    key3 = keys[3],
                    key4 = keys[4],
                    key5 = keys[5],
                    value0 = values[0],
                    value1 = values[1],
                    value2 = values[2],
                    value3 = values[3],
                    value4 = values[4],
                    value5 = values[5]
                )

            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                if (depth + 1 == ngram.length) {
                    val key = ngram[depth]
                    if (key <= key0) {
                        if (key == key0) return value0
                    } else if (key >= key5) {
                        if (key == key5) return value5
                    } else {
                        if (key1 == key) return value1
                        if (key2 == key) return value2
                        if (key3 == key) return value3
                        if (key4 == key) return value4
                    }
                }
                return 0F
            }
        }

        /**
         * A node with only 7 leaf children
         */
        private data class PreLeaf7(
            override val frequency: Float,
            private val key0: Char,
            private val key1: Char,
            private val key2: Char,
            private val key3: Char,
            private val key4: Char,
            private val key5: Char,
            private val key6: Char,
            private val value0: Float,
            private val value1: Float,
            private val value2: Float,
            private val value3: Float,
            private val value4: Float,
            private val value5: Float,
            private val value6: Float
        ) : RelativeFrequencies() {

            constructor(frequency: Float, keys: CharArray, values: FloatArray) :
                this(
                    frequency = frequency,
                    key0 = keys[0],
                    key1 = keys[1],
                    key2 = keys[2],
                    key3 = keys[3],
                    key4 = keys[4],
                    key5 = keys[5],
                    key6 = keys[6],
                    value0 = values[0],
                    value1 = values[1],
                    value2 = values[2],
                    value3 = values[3],
                    value4 = values[4],
                    value5 = values[5],
                    value6 = values[6]
                )

            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                if (depth + 1 == ngram.length) {
                    val key = ngram[depth]
                    if (key <= key0) {
                        if (key == key0) return value0
                    } else if (key >= key6) {
                        if (key == key6) return value6
                    } else if (key < key3) {
                        if (key1 == key) return value1
                        if (key2 == key) return value2
                    } else if (key > key3) {
                        if (key4 == key) return value4
                        if (key5 == key) return value5
                    } else return value3
                }
                return 0F
            }
        }

        /**
         * A node with only leaf children
         */
        private class PreLeafN(
            override val frequency: Float,
            private val keys: CharArray,
            private val values: FloatArray
        ) : RelativeFrequencies() {
            override fun getImpl(ngram: CharSequence, depth: Int): Float {
                if (depth == ngram.length) return frequency
                if (depth + 1 == ngram.length) {
                    val key = ngram[depth]
                    val i = keys.binarySearch(key)
                    if (i >= 0) return values[i]
                }
                return 0F
            }
        }

        /**
         * A leaf node
         */
        private data class Leaf(override val frequency: Float) : RelativeFrequencies() {
            override fun getImpl(ngram: CharSequence, depth: Int): Float =
                if (depth == ngram.length) frequency else 0F
        }

        private class MutableNode(
            var frequency: Float = 0F,
            val children: TreeMap<Char, MutableNode> = TreeMap()
        ) {
            operator fun set(ngram: String, frequency: Float) {
                var node = this
                repeat(ngram.length) { i ->
                    node = node.children.computeIfAbsent(ngram[i]) { MutableNode() }
                }
                node.frequency = frequency
            }

            private class BuilderCache(
                val keysCache: MutableMap<List<Char>, CharArray> = HashMap(128),
                val nodeCache: MutableMap<RelativeFrequencies, RelativeFrequencies> = HashMap(256)
            )

            fun toRelativeFrequencies(): RelativeFrequencies = toRelativeFrequencies(BuilderCache())

            private fun toRelativeFrequencies(builderCache: BuilderCache): RelativeFrequencies {
                if (children.isEmpty()) {
                    // leaf node
                    val node = Leaf(frequency)
                    return builderCache.nodeCache.computeIfAbsent(node) { node }
                }

                var keysArray = children.keys.toCharArray()

                if (children.values.all { it.children.isEmpty() }) {
                    // pre-leafs node
                    if (children.size == 1) {
                        val node = PreLeaf1(frequency, keysArray.single(), children.values.first().frequency)
                        return builderCache.nodeCache.computeIfAbsent(node) { node }
                    }

                    val valuesArray = FloatArray(children.size)
                    children.values.forEachIndexed { index, mutableNode ->
                        valuesArray[index] = mutableNode.frequency
                    }

                    if (children.size == 2) return PreLeaf2(frequency, keysArray, valuesArray)
                    if (children.size == 3) return PreLeaf3(frequency, keysArray, valuesArray)
                    if (children.size == 4) return PreLeaf4(frequency, keysArray, valuesArray)
                    if (children.size == 5) return PreLeaf5(frequency, keysArray, valuesArray)
                    if (children.size == 6) return PreLeaf6(frequency, keysArray, valuesArray)
                    if (children.size == 7) return PreLeaf7(frequency, keysArray, valuesArray)

                    if (keysArray.size < 32) {
                        keysArray = builderCache.keysCache.computeIfAbsent(keysArray.asList()) { keysArray }
                    }
                    return PreLeafN(frequency = frequency, keys = keysArray, values = valuesArray)
                }

                // intermediate node
                if (keysArray.size < 32) {
                    keysArray = builderCache.keysCache.computeIfAbsent(keysArray.asList()) { keysArray }
                }

                val valuesArray = arrayOfNulls<RelativeFrequencies>(children.size)
                children.values.forEachIndexed { index, mutableNode ->
                    valuesArray[index] = mutableNode.toRelativeFrequencies(builderCache)
                }

                @Suppress("UNCHECKED_CAST")
                return GenericNode(
                    frequency = frequency,
                    keys = keysArray,
                    values = valuesArray as Array<RelativeFrequencies>
                )
            }
        }

        companion object {
            internal fun build(relativeFrequencies: Sequence<Pair<String, Float>>): RelativeFrequencies {
                val mutableRoot = MutableNode()
                for ((ngram, frequency) in relativeFrequencies) {
                    mutableRoot[ngram] = frequency
                }
                return mutableRoot.toRelativeFrequencies()
            }
        }
    }
}
