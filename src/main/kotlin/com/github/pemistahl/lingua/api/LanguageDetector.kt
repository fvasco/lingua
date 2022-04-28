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

package com.github.pemistahl.lingua.api

import com.github.pemistahl.lingua.api.Language.*
import com.github.pemistahl.lingua.internal.*
import com.github.pemistahl.lingua.internal.Constant.CHARS_TO_LANGUAGES_MAPPING
import com.github.pemistahl.lingua.internal.Constant.MULTIPLE_WHITESPACE
import com.github.pemistahl.lingua.internal.Constant.NO_LETTER
import com.github.pemistahl.lingua.internal.Constant.NUMBERS
import com.github.pemistahl.lingua.internal.Constant.PUNCTUATION
import com.github.pemistahl.lingua.internal.Constant.isJapaneseAlphabet
import com.github.pemistahl.lingua.internal.util.SubSequence
import com.github.pemistahl.lingua.internal.util.extension.incrementCounter
import com.github.pemistahl.lingua.internal.util.extension.isLogogram
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.decodeFromStream
import java.util.*
import java.util.concurrent.Callable
import java.util.concurrent.ForkJoinPool
import kotlin.math.ln

/**
 * Detects the language of given input text.
 */
class LanguageDetector internal constructor(
    internal val languages: Set<Language>,
    internal val minimumRelativeDistance: Double,
    private val isEveryLanguageModelPreloaded: Boolean
) {

    internal val languageModels = EnumMap<Language, TrainingDataLanguageModel>(Language::class.java)

    internal val numberOfLoadedLanguages: Int get() = languages.size

    private val languagesWithUniqueCharacters = languages.filterNot { it.uniqueCharacters.isNullOrBlank() }.asSequence()
    private val oneLanguageAlphabets = Alphabet.allSupportingExactlyOneLanguage().filterValues {
        it in languages
    }

    init {
        if (isEveryLanguageModelPreloaded) {
            preloadLanguageModels()
        }
    }

    /**
     * Detects the language of given input text.
     *
     * @param text The input text to detect the language for.
     * @return The identified language or [Language.UNKNOWN].
     * @throws IllegalStateException If [destroy] has been invoked before on this instance of [LanguageDetector].
     */
    fun detectLanguageOf(text: String): Language {
        val confidenceValues = computeLanguageConfidenceValues(text)

        if (confidenceValues.isEmpty()) return UNKNOWN

        val mostLikelyLanguage = confidenceValues.firstKey()
        if (confidenceValues.size == 1) return mostLikelyLanguage

        val mostLikelyLanguageProbability = confidenceValues.getValue(mostLikelyLanguage)
        val secondMostLikelyLanguageProbability = confidenceValues.values.elementAt(1)

        return when {
            mostLikelyLanguageProbability == secondMostLikelyLanguageProbability -> UNKNOWN
            (mostLikelyLanguageProbability - secondMostLikelyLanguageProbability) < minimumRelativeDistance -> UNKNOWN
            else -> mostLikelyLanguage
        }
    }

    /**
     * Computes confidence values for every language considered possible for the given input text.
     *
     * The values that this method computes are part of a **relative** confidence metric, not of an absolute one.
     * Each value is a number between 0.0 and 1.0. The most likely language is always returned with value 1.0.
     * All other languages get values assigned which are lower than 1.0, denoting how less likely those languages
     * are in comparison to the most likely language.
     *
     * The map returned by this method does not necessarily contain all languages which the calling instance of
     * [LanguageDetector] was built from. If the rule-based engine decides that a specific language is truly impossible,
     * then it will not be part of the returned map. Likewise, if no ngram probabilities can be found within the
     * detector's languages for the given input text, the returned map will be empty. The confidence value for
     * each language not being part of the returned map is assumed to be 0.0.
     *
     * @param text The input text to detect the language for.
     * @return A map of all possible languages, sorted by their confidence value in descending order.
     */
    fun computeLanguageConfidenceValues(text: String): SortedMap<Language, Double> {
        val values = TreeMap<Language, Double>()
        val cleanedUpText = cleanUpInputText(text)

        if (cleanedUpText.isEmpty() || NO_LETTER.matches(cleanedUpText)) return values

        val words = splitTextIntoWords(cleanedUpText)
        val languageDetectedByRules = detectLanguageWithRules(words)

        if (languageDetectedByRules != UNKNOWN) {
            values[languageDetectedByRules] = 1.0
            return values
        }

        val filteredLanguages = filterLanguagesByRules(words)

        // single language optimization
        filteredLanguages.singleOrNull()?.let { filteredLanguage ->
            values[filteredLanguage] = 1.0
            return values
        }

        val ngramSizeRange = if (cleanedUpText.length >= 120) (3..3) else (1..5)
        val allProbabilitiesAndUnigramCounts = ngramSizeRange.filter { i -> cleanedUpText.length >= i }.map { i ->
            val testDataModel = TestDataLanguageModel.fromText(cleanedUpText, ngramLength = i)
            val probabilities = computeLanguageProbabilities(testDataModel, filteredLanguages)

            val unigramCounts =
                if (i != 1 || probabilities.isEmpty()) {
                    emptyMap()
                } else {
                    countUnigramsOfInputText(testDataModel, filteredLanguages.intersect(languages))
                }

            Pair(probabilities, unigramCounts)
        }

        val allProbabilities = allProbabilitiesAndUnigramCounts.map { (probabilities, _) -> probabilities }
        val unigramCounts = allProbabilitiesAndUnigramCounts[0].second
        val summedUpProbabilities = sumUpProbabilities(allProbabilities, unigramCounts, filteredLanguages)
        val highestProbability = summedUpProbabilities.maxByOrNull { it.value }?.value ?: return sortedMapOf()
        val confidenceValues = summedUpProbabilities.mapValues { highestProbability / it.value }
        val sortedByConfidenceValue = compareByDescending<Language> { language -> confidenceValues[language] }
        val sortedByConfidenceValueThenByLanguage = sortedByConfidenceValue.thenBy { language -> language }

        return confidenceValues.toSortedMap(sortedByConfidenceValueThenByLanguage)
    }

    /**
     * This fork does not allocate native resources
     */
    @Deprecated("This fork does not allocate native resources", replaceWith = ReplaceWith(""))
    fun destroy() {
    }

    internal fun cleanUpInputText(text: String): String {
        return text.trim().lowercase()
            .replace(PUNCTUATION, "")
            .replace(NUMBERS, "")
            .replace(MULTIPLE_WHITESPACE, " ")
    }

    internal fun splitTextIntoWords(text: String): List<CharSequence> {
        val words = mutableListOf<CharSequence>()
        var nextWordStart = 0
        for (i in text.indices) {
            val char = text[i]

            if (char == ' ') {
                if (nextWordStart != i) {
                    words.add(SubSequence(text, nextWordStart, i - nextWordStart))
                }
                nextWordStart = i + 1
            } else if (char.isLogogram()) {
                if (nextWordStart != i) {
                    words.add(SubSequence(text, nextWordStart, i - nextWordStart))
                }

                words.add(text[i].toString())
                nextWordStart = i + 1
            }
        }

        if (nextWordStart != text.length) {
            words.add(SubSequence(text, nextWordStart, text.length - nextWordStart))
        }
        return words
    }

    internal fun countUnigramsOfInputText(
        unigramLanguageModel: TestDataLanguageModel,
        filteredLanguages: Set<Language>
    ): Map<Language, Int> {
        val unigramCounts = mutableMapOf<Language, Int>()
        for (language in filteredLanguages) {
            for (unigram in unigramLanguageModel.ngrams) {
                val probability = lookUpNgramProbability(language, unigram)
                if (probability > 0) {
                    unigramCounts.incrementCounter(language)
                }
            }
        }
        return unigramCounts
    }

    internal fun sumUpProbabilities(
        probabilities: List<Map<Language, Double>>,
        unigramCountsOfInputText: Map<Language, Int>,
        filteredLanguages: Set<Language>
    ): Map<Language, Double> {
        val summedUpProbabilities = mutableMapOf<Language, Double>()
        for (language in filteredLanguages) {
            summedUpProbabilities[language] = probabilities.sumOf { it[language] ?: 0.0 }

            if (unigramCountsOfInputText.containsKey(language)) {
                summedUpProbabilities[language] = summedUpProbabilities.getValue(language) /
                    unigramCountsOfInputText.getValue(language)
            }
        }
        return summedUpProbabilities.filter { it.value != 0.0 }
    }

    internal fun detectLanguageWithRules(words: List<CharSequence>): Language {
        val totalLanguageCounts = mutableMapOf<Language, Int>()

        for (word in words) {
            val wordLanguageCounts = mutableMapOf<Language, Int>()

            for (character in word) {
                var isMatch = false
                for ((alphabet, language) in oneLanguageAlphabets) {
                    if (alphabet.matches(character)) {
                        wordLanguageCounts.incrementCounter(language)
                        isMatch = true
                        break
                    }
                }
                if (!isMatch) {
                    when {
                        Alphabet.HAN.matches(character) -> wordLanguageCounts.incrementCounter(CHINESE)
                        isJapaneseAlphabet(character) -> wordLanguageCounts.incrementCounter(JAPANESE)
                        Alphabet.LATIN.matches(character) ||
                            Alphabet.CYRILLIC.matches(character) ||
                            Alphabet.DEVANAGARI.matches(character) ->
                            languagesWithUniqueCharacters.filter {
                                it.uniqueCharacters?.contains(character) ?: false
                            }.forEach {
                                wordLanguageCounts.incrementCounter(it)
                            }
                    }
                }
            }

            if (wordLanguageCounts.isEmpty()) {
                totalLanguageCounts.incrementCounter(UNKNOWN)
            } else if (wordLanguageCounts.size == 1) {
                val language = wordLanguageCounts.keys.first()
                if (language in languages) {
                    totalLanguageCounts.incrementCounter(language)
                } else {
                    totalLanguageCounts.incrementCounter(UNKNOWN)
                }
            } else {
                val sortedWordLanguageCounts = wordLanguageCounts.toList().sortedByDescending { it.second }
                val (mostFrequentLanguage, firstCharCount) = sortedWordLanguageCounts[0]
                val (_, secondCharCount) = sortedWordLanguageCounts[1]

                if (firstCharCount > secondCharCount && mostFrequentLanguage in languages) {
                    totalLanguageCounts.incrementCounter(mostFrequentLanguage)
                } else {
                    totalLanguageCounts.incrementCounter(UNKNOWN)
                }
            }
        }

        val unknownLanguageCount = totalLanguageCounts[UNKNOWN] ?: 0
        if (unknownLanguageCount < (0.5 * words.size)) {
            totalLanguageCounts.remove(UNKNOWN)
        }

        if (totalLanguageCounts.isEmpty()) {
            return UNKNOWN
        }
        if (totalLanguageCounts.size == 1) {
            return totalLanguageCounts.keys.first()
        }
        if (totalLanguageCounts.size == 2 &&
            totalLanguageCounts.containsKey(CHINESE) &&
            totalLanguageCounts.containsKey(JAPANESE)
        ) {
            return JAPANESE
        }
        val sortedTotalLanguageCounts = totalLanguageCounts.toList().sortedByDescending { it.second }
        val (mostFrequentLanguage, firstCharCount) = sortedTotalLanguageCounts[0]
        val (_, secondCharCount) = sortedTotalLanguageCounts[1]

        return if (firstCharCount == secondCharCount) UNKNOWN else mostFrequentLanguage
    }

    internal fun filterLanguagesByRules(words: List<CharSequence>): Set<Language> {
        val detectedAlphabets = mutableMapOf<Alphabet, Int>()

        for (word in words) {
            for (alphabet in Alphabet.values()) {
                if (alphabet.matches(word)) {
                    detectedAlphabets.incrementCounter(alphabet)
                    break
                }
            }
        }

        if (detectedAlphabets.isEmpty()) {
            return languages
        }

        if (detectedAlphabets.size > 1) {
            val distinctAlphabets = mutableSetOf<Int>()
            for (count in detectedAlphabets.values) {
                distinctAlphabets.add(count)
            }
            if (distinctAlphabets.size == 1) {
                return languages
            }
        }

        val mostFrequentAlphabet = detectedAlphabets.entries.maxByOrNull { it.value }!!.key
        val filteredLanguages = languages.filterTo(mutableSetOf()) { it.alphabets.contains(mostFrequentAlphabet) }
        val languageCounts = mutableMapOf<Language, Int>()

        for (word in words) {
            for ((characters, languages) in CHARS_TO_LANGUAGES_MAPPING) {
                for (character in characters) {
                    if (character in word) {
                        for (language in languages) {
                            if (filteredLanguages.contains(language)) {
                                languageCounts.incrementCounter(language)
                            }
                        }
                    }
                }
            }
        }

        val languagesSubset = languageCounts.filterValues { it >= words.size / 2.0 }.keys

        return if (languagesSubset.isNotEmpty()) {
            filteredLanguages.intersect(languagesSubset)
        } else {
            filteredLanguages
        }
    }

    internal fun computeLanguageProbabilities(
        testDataModel: TestDataLanguageModel,
        filteredLanguages: Set<Language>
    ): Map<Language, Double> {
        val probabilities = mutableMapOf<Language, Double>()
        for (language in filteredLanguages) {
            probabilities[language] = computeSumOfNgramProbabilities(language, testDataModel.ngrams)
        }
        return probabilities.filter { it.value < 0.0 }
    }

    internal fun computeSumOfNgramProbabilities(
        language: Language,
        ngrams: Set<Ngram>
    ): Double {
        var probabilitiesSum = 0.0

        for (ngram in ngrams) {
            for (elem in ngram.rangeOfLowerOrderNgrams()) {
                val probability = lookUpNgramProbability(language, elem)
                if (probability > 0) {
                    probabilitiesSum += ln(probability.toDouble())
                    break
                }
            }
        }
        return probabilitiesSum
    }

    internal fun lookUpNgramProbability(
        language: Language,
        ngram: Ngram
    ): Float {
        require(ngram.length > 0) { "Zerogram detected" }
        require(ngram.length <= 5) { "unsupported ngram length detected: ${ngram.length}" }
        return loadLanguageModels(language).getRelativeFrequency(ngram)
    }

    private fun loadLanguageModels(language: Language): TrainingDataLanguageModel =
        if (isEveryLanguageModelPreloaded) languageModels.getValue(language)
        else synchronized(languageModels) {
            languageModels.computeIfAbsent(language) {
                loadLanguageModel(language)
            }
        }

    private fun preloadLanguageModels() {
        val threadPool = ForkJoinPool.commonPool()
        languages.map { language ->
            language to threadPool.submit(Callable { loadLanguageModel(language) })
        }.forEach { (language, future) ->
            languageModels[language] = future.get()
        }
    }

    override fun equals(other: Any?) = when {
        this === other -> true
        other !is LanguageDetector -> false
        languages != other.languages -> false
        minimumRelativeDistance != other.minimumRelativeDistance -> false
        else -> true
    }

    override fun hashCode() = 31 * languages.hashCode() + minimumRelativeDistance.hashCode()

    private companion object {
        private val modelCache = HashMap<Language, TrainingDataLanguageModel>()

        private fun loadLanguageModel(language: Language): TrainingDataLanguageModel {
            synchronized(modelCache) {
                modelCache[language]?.let { return it }
            }
            val jsonLanguageModels: Sequence<JsonLanguageModel> = (1..5).asSequence().mapNotNull { ngramLength ->
                val filePath =
                    "/language-models/${language.isoCode639_1}/${Ngram.getNgramNameByLength(ngramLength)}s.json"
                Language::class.java.getResourceAsStream(filePath)
                    ?.use { Json.decodeFromStream(JsonLanguageModel.serializer(), it) }
            }
            val model = TrainingDataLanguageModel.fromJson(language, jsonLanguageModels)
            synchronized(modelCache) {
                modelCache.putIfAbsent(language, model)
                return modelCache.getValue(language)
            }
        }
    }
}
